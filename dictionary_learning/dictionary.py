"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
from huggingface_hub import PyTorchModelHubMixin

import torch as th
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu
import einops
from warnings import warn
from typing import Callable
from enum import Enum, auto
from .utils import set_decoder_norm_to_unit_norm


class Dictionary(ABC, nn.Module, PyTorchModelHubMixin):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, path, from_hub=False, device=None, dtype=None, **kwargs
    ) -> "Dictionary":
        """
        Load a pretrained dictionary from a file or hub.

        Args:
            path: Path to local file or hub model id
            from_hub: If True, load from HuggingFace hub using PyTorchModelHubMixin
            device: Device to load the model to
            **kwargs: Additional arguments passed to loading function
        """
        model = super(Dictionary, cls).from_pretrained(path, **kwargs)
        if device is not None:
            model.to(device)
        if dtype is not None:
            model.to(dtype=dtype)
        return model


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(th.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = th.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = th.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    @classmethod
    def from_pretrained(
        cls, path, dtype=th.float, from_hub=False, device=None, **kwargs
    ):
        if from_hub:
            return super().from_pretrained(path, dtype=dtype, device=device, **kwargs)

        # Existing custom loading logic
        state_dict = th.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(dtype=dtype, device=device)
        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x

    @classmethod
    def from_pretrained(cls, path, dtype=th.float, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        return cls(None)


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(
        self, activation_dim, dict_size, initialization="default", device=None
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(th.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(th.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(th.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(th.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = th.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, return_gate=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f):
        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    @classmethod
    def from_pretrained(cls, path, from_hub=False, device=None, dtype=None, **kwargs):
        if from_hub:
            return super().from_pretrained(path, device=device, dtype=dtype, **kwargs)

        # Existing custom loading logic
        state_dict = th.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(th.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(th.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(th.empty(dict_size, activation_dim, device=device))
        self.b_dec = nn.Parameter(th.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(th.zeros(dict_size, device=device))

        self.apply_b_dec_to_input = False

        # rows of decoder weight matrix are initialized to unit vectors
        self.W_enc.data = th.randn_like(self.W_enc)
        self.W_enc.data = self.W_enc / self.W_enc.norm(dim=0, keepdim=True)
        self.W_dec.data = self.W_enc.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))
        f = f * self.W_dec.norm(dim=1)

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        f = f / self.W_dec.norm(dim=1)
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        from_hub: bool = False,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            if from_hub:
                return super().from_pretrained(
                    path, device=device, dtype=dtype, **kwargs
                )
            state_dict = th.load(path)
            dict_size, activation_dim = state_dict["W_enc"].shape
            autoencoder = cls(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
        else:
            from sae_lens import SAE

            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            assert (
                cfg_dict["finetuning_scaling_factor"] == False
            ), "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)


class BatchTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", th.tensor(k, dtype=th.int))
        self.register_buffer("threshold", th.tensor(-1.0, dtype=th.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(th.zeros(activation_dim))

    def encode(
        self, x: th.Tensor, return_active: bool = False, use_threshold: bool = True
    ):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts_BF = (
                th.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: th.Tensor) -> th.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: th.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(
        cls, path, k=None, device=None, from_hub=False, **kwargs
    ) -> "BatchTopKSAE":
        if from_hub:
            return super().from_pretrained(path, device=device, **kwargs)

        state_dict = th.load(path, weights_only=True)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


# TODO merge this with AutoEncoder
class AutoEncoderNew(Dictionary, nn.Module):
    """
    The autoencoder architecture and initialization used in https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = th.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else:  # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f

    @classmethod
    def from_pretrained(cls, path, device=None, from_hub=False, dtype=None, **kwargs):
        if from_hub:
            return super().from_pretrained(path, device=device, dtype=dtype, **kwargs)

        state_dict = th.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class CrossCoderEncoder(nn.Module):
    """
    A crosscoder encoder
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers=None,
        same_init_for_all_layers: bool = False,
        norm_init_scale: float | None = None,
        encoder_layers: list[int] | None = None,
    ):
        super().__init__()

        if encoder_layers is None:
            if num_layers is None:
                raise ValueError(
                    "Either encoder_layers or num_layers must be specified"
                )
            encoder_layers = list(range(num_layers))
        else:
            num_layers = len(encoder_layers)
        self.encoder_layers = encoder_layers
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        if same_init_for_all_layers:
            weight = init.kaiming_uniform_(th.empty(activation_dim, dict_size))
            weight = weight.repeat(num_layers, 1, 1)
        else:
            weight = init.kaiming_uniform_(
                th.empty(num_layers, activation_dim, dict_size)
            )
        if norm_init_scale is not None:
            weight = weight / weight.norm(dim=1, keepdim=True) * norm_init_scale
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(th.zeros(dict_size))

    def forward(
        self,
        x: th.Tensor,
        return_no_sum: bool = False,
        select_features: list[int] | None = None,
        **kwargs,
    ) -> th.Tensor:  # (batch_size, activation_dim)
        """
        Convert activations to features for each layer

        Args:
            x: (batch_size, n_layers, activation_dim)
        Returns:
            f: (batch_size, dict_size)
        """
        x = x[:, self.encoder_layers]
        if select_features is not None:
            w = self.weight[:, :, select_features]
            bias = self.bias[select_features]
        else:
            w = self.weight
            bias = self.bias
        f = th.einsum("bld, ldf -> blf", x, w)
        if not return_no_sum:
            return relu(f.sum(dim=1) + bias)
        else:
            return relu(f.sum(dim=1) + bias), relu(f + bias)


class CrossCoderDecoder(nn.Module):
    """
    A crosscoder decoder
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers: bool = False,
        norm_init_scale: float | None = None,
        init_with_weight: th.Tensor | None = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.bias = nn.Parameter(th.zeros(num_layers, activation_dim))
        if init_with_weight is not None:
            self.weight = nn.Parameter(init_with_weight)
        else:
            if same_init_for_all_layers:
                weight = init.kaiming_uniform_(th.empty(dict_size, activation_dim))
                weight = weight.repeat(num_layers, 1, 1)
            else:
                weight = init.kaiming_uniform_(
                    th.empty(num_layers, dict_size, activation_dim)
                )
            if norm_init_scale is not None:
                weight = weight / weight.norm(dim=2, keepdim=True) * norm_init_scale
            self.weight = nn.Parameter(weight)

    def forward(
        self,
        f: th.Tensor,
        select_features: list[int] | None = None,
        add_bias: bool = True,
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        # f: (batch_size, n_layers, dict_size)
        """
        Convert features to activations for each layer

        Args:
            f: (batch_size, dict_size) or (batch_size, n_layers, dict_size)
        Returns:
            x: (batch_size, n_layers, activation_dim)
        """
        if select_features is not None:
            w = self.weight[:, select_features]
        else:
            w = self.weight
        if f.dim() == 2:
            x = th.einsum("bf, lfd -> bld", f, w)
        else:
            x = th.einsum("blf, lfd -> bld", f, w)
        if add_bias:
            x += self.bias
        return x


class CodeNormalization(Enum):
    """
    Enumeration of supported normalization for dictionary learning.

    Attributes:
        CROSSCODER: Sum of norms of the decoder rows for each layer
        SAE: Norm of the concatenated decoder rows (equivalent to SAE on the concatenated activations)
        MIXED: Sum of SAE and CC losses
        DECOUPLED: Norm of the decoder rows for each layer (no sum)

    """

    CROSSCODER = auto()
    SAE = auto()
    MIXED = auto()
    NONE = auto()
    DECOUPLED = auto()

    @classmethod
    def from_string(cls, code_norm_type_str: str) -> "CodeNormalization":
        """
        Initialize a CodeNormalization from a string representation.

        Args:
            code_norm_type_str: String representation of the code normalization type

        Returns:
            The corresponding CodeNormalization enum value

        Raises:
            ValueError: If the string does not match any CodeNormalization
        """
        try:
            return cls[code_norm_type_str.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown code normalization type: {code_norm_type_str}. Available types: {[lt.name for lt in cls]}"
            )

    def __str__(self) -> str:
        """
        String representation of the LossType.

        Returns:
            The name of the loss type in uppercase
        """
        return self.name

    def __repr__(self) -> str:
        """
        String representation of the LossType.

        Returns:
            The name of the loss type in uppercase
        """
        return self.name


class CrossCoder(Dictionary, nn.Module):
    """
        A crosscoder using the AutoEncoderNew architecture for two models.
    pl
        encoder: shape (num_layers, activation_dim, dict_size)
        decoder: shape (num_layers, dict_size, activation_dim)
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers=False,
        norm_init_scale: float | None = None, 
        init_with_transpose=True,
        encoder_layers: list[int] | None = None,
        latent_processor: Callable | None = None,
        num_decoder_layers: int | None = None,
        code_normalization: CodeNormalization | str = CodeNormalization.CROSSCODER,
        code_normalization_alpha_sae: float | None = 1.0,
        code_normalization_alpha_cc: float | None = 0.1,
    ):
        """
        Args:
            same_init_for_all_layers: if True, initialize all layers with the same vector
            norm_init_scale: if not None, initialize the weights with a norm of this value
            init_with_transpose: if True, initialize the decoder weights with the transpose of the encoder weights
            latent_processor: Function to process the latents after encoding
            num_decoder_layers: Number of decoder layers. If None, use num_layers.
            code_normalization: Sparsity loss type to use for the crosscoder
            code_normalization_alpha_sae: Weight of SAE loss for the sparsity loss MIXED
            code_normalization_alpha_cc: Weight of CC loss for the sparsity loss MIXED
        """
        super().__init__()
        if num_decoder_layers is None:
            num_decoder_layers = num_layers

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.latent_processor = latent_processor
        if isinstance(code_normalization, str):
            code_normalization = CodeNormalization.from_string(code_normalization)
        else:
            self._hub_mixin_config["code_normalization"] = code_normalization.name
        self.code_normalization = code_normalization
        self.code_normalization_alpha_sae = code_normalization_alpha_sae
        self.code_normalization_alpha_cc = code_normalization_alpha_cc
        self.encoder = CrossCoderEncoder(
            activation_dim,
            dict_size,
            num_layers,
            same_init_for_all_layers=same_init_for_all_layers,
            norm_init_scale=norm_init_scale,
            encoder_layers=encoder_layers,
        )

        if init_with_transpose:
            decoder_weight = einops.rearrange(
                self.encoder.weight.data.clone(),
                "num_layers activation_dim dict_size -> num_layers dict_size activation_dim",
            )
        else:
            decoder_weight = None
        self.decoder = CrossCoderDecoder(
            activation_dim,
            dict_size,
            num_decoder_layers,
            same_init_for_all_layers=same_init_for_all_layers,
            init_with_weight=decoder_weight,
            norm_init_scale=norm_init_scale,
        )
        self.register_buffer(
            "code_normalization_id", th.tensor(code_normalization.value)
        )
        self.decoupled_code = self.code_normalization == CodeNormalization.DECOUPLED

    def get_code_normalization(
        self, select_features: list[int] | None = None
    ) -> th.Tensor:
        if select_features is not None:
            dw = self.decoder.weight[:, select_features]
        else:
            dw = self.decoder.weight

        if self.code_normalization == CodeNormalization.SAE:
            weight_norm = dw.norm(dim=(0, 2)).unsqueeze(0)
        elif self.code_normalization == CodeNormalization.MIXED:
            weight_norm_sae = dw.norm(dim=(0, 2)).unsqueeze(0)
            weight_norm_cc = dw.norm(dim=2).sum(dim=0, keepdim=True)
            weight_norm = (
                weight_norm_sae * self.code_normalization_alpha_sae
                + weight_norm_cc * self.code_normalization_alpha_cc
            )
        elif self.code_normalization == CodeNormalization.NONE:
            weight_norm = th.tensor(1.0)
        elif self.code_normalization == CodeNormalization.CROSSCODER:
            weight_norm = dw.norm(dim=2).sum(dim=0, keepdim=True)
        elif self.code_normalization == CodeNormalization.DECOUPLED:
            weight_norm = dw.norm(dim=2)
        else:
            raise NotImplementedError(
                f"Code normalization {self.code_normalization} not implemented"
            )
        return weight_norm

    def encode(
        self, x: th.Tensor, **kwargs
    ) -> th.Tensor:  # (batch_size, n_layers, dict_size)
        # x: (batch_size, n_layers, activation_dim)
        return self.encoder(x, **kwargs)

    def get_activations(
        self, x: th.Tensor, use_threshold: bool = True, select_features=None, **kwargs
    ):
        f = self.encode(x, use_threshold=use_threshold, **kwargs)
        weight_norm = self.get_code_normalization()
        if self.decoupled_code:
            weight_norm = weight_norm.sum(dim=0, keepdim=True)
        if select_features is not None:
            return (f * weight_norm)[:, select_features]
        return f * weight_norm

    def decode(
        self, f: th.Tensor, **kwargs
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        # f: (batch_size, n_layers, dict_size)
        return self.decoder(f, **kwargs)

    def forward(self, x: th.Tensor, output_features=False):
        """
        Forward pass of the crosscoder.
        x : activations to be encoded and decoded
        output_features : if True, return the encoded features as well as the decoded x
        """
        f = self.encode(x)
        if self.latent_processor is not None:
            f = self.latent_processor(f)
        x_hat = self.decode(f)

        if output_features:
            # Scale features by decoder column norms
            weight_norm = self.get_code_normalization()
            if self.decoupled_code:
                weight_norm = weight_norm.sum(dim=0, keepdim=True)
            return x_hat, f * weight_norm
        else:
            return x_hat

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
        from_hub: bool = False,
        code_normalization: CodeNormalization | str | None = None,
        **kwargs,
    ):
        """
        Load a pretrained crosscoder from a file.
        """
        if isinstance(code_normalization, str):
            code_normalization = CodeNormalization.from_string(code_normalization)
        if from_hub:
            return super().from_pretrained(path, device=device, dtype=dtype, **kwargs)

        state_dict = th.load(path, map_location="cpu", weights_only=True)
        if "encoder.weight" not in state_dict:
            warn(
                "crosscoder state dict was saved while torch.compiled was enabled. Fixing..."
            )
            state_dict = {k.split("_orig_mod.")[1]: v for k, v in state_dict.items()}
        if "code_normalization_id" not in state_dict:
            if code_normalization is None:
                warn(
                    "No code normalization id found in {path}. This is likely due to saving the model using an older version of dictionary_learning. Assuming code_normalization is CROSSCODER, if not pass code_normalization as a from_pretrained kwarg"
                )
                state_dict["code_normalization_id"] = th.tensor(
                    CodeNormalization.CROSSCODER.value, dtype=th.int
                )
            else:
                state_dict["code_normalization_id"] = th.tensor(
                    code_normalization.value, dtype=th.int
                )
        num_layers, activation_dim, dict_size = state_dict["encoder.weight"].shape
        crosscoder = cls(
            activation_dim,
            dict_size,
            num_layers,
            code_normalization=CodeNormalization._value2member_map_[
                state_dict["code_normalization_id"].item()
            ],
        )
        crosscoder.load_state_dict(state_dict)

        if device is not None:
            crosscoder = crosscoder.to(device)
        return crosscoder.to(dtype=dtype)

    def resample_neurons(self, deads, activations):
        # https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling
        # compute loss for each activation
        losses = (
            (activations - self.forward(activations)).norm(dim=-1).mean(dim=-1).square()
        )

        # sample input to create encoder/decoder weights from
        n_resample = min([deads.sum(), losses.shape[0]])
        print("Resampling", n_resample, "neurons")
        indices = th.multinomial(losses, num_samples=n_resample, replacement=False)
        sampled_vecs = activations[indices]  # (n_resample, num_layers, activation_dim)

        # get norm of the living neurons
        # encoder.weight: (num_layers, activation_dim, dict_size)
        # decoder.weight: (num_layers, dict_size, activation_dim)
        alive_norm = self.encoder.weight[:, :, ~deads].norm(dim=-2)
        alive_norm = alive_norm.mean(dim=-1)  # (num_layers)
        # convert to (num_layers, 1, 1)
        alive_norm = einops.repeat(alive_norm, "num_layers -> num_layers 1 1")

        # resample first n_resample dead neurons
        deads[deads.nonzero()[n_resample:]] = False
        self.encoder.weight[:, :, deads] = (
            sampled_vecs.permute(1, 2, 0) * alive_norm * 0.05
        )
        sampled_vecs = sampled_vecs.permute(1, 0, 2)
        self.decoder.weight[:, deads, :] = th.nn.functional.normalize(
            sampled_vecs, dim=-1
        )
        self.encoder.bias[deads] = 0.0


class BatchTopKCrossCoder(CrossCoder):
    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        k: int | th.Tensor = 100,
        norm_init_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            activation_dim,
            dict_size,
            num_layers,
            norm_init_scale=norm_init_scale,
            *args,
            **kwargs,
        )
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers

        if not isinstance(k, th.Tensor):
            k = th.tensor(k, dtype=th.int)

        self.register_buffer("k", k)
        threshold = [-1.0] * num_layers if self.decoupled_code else -1.0
        self.register_buffer("threshold", th.tensor(threshold, dtype=th.float32))

    def encode(
        self,
        x: th.Tensor,
        return_active: bool = False,
        use_threshold: bool = True,
        select_features: list[int] | None = None,
    ):
        if self.decoupled_code:
            return self.encode_decoupled(
                x, return_active, use_threshold, select_features
            )
        batch_size = x.size(0)
        post_relu_f = super().encode(x, select_features=select_features)
        code_normalization = self.get_code_normalization(select_features)
        post_relu_f_scaled = post_relu_f * code_normalization
        if use_threshold:
            f = post_relu_f * (post_relu_f_scaled > self.threshold)
        else:
            # Flatten and perform batch top-k
            flattened_acts_scaled = post_relu_f_scaled.flatten()
            post_topk = flattened_acts_scaled.topk(
                self.k * batch_size, sorted=False, dim=-1
            )
            post_topk_values = post_relu_f.flatten()[post_topk.indices]
            f = (
                th.zeros_like(flattened_acts_scaled)
                .scatter_(-1, post_topk.indices, post_topk_values)
                .reshape(post_relu_f.shape)
            )
        if return_active:
            return (
                f,
                f * code_normalization,
                f.sum(0) > 0,
                post_relu_f,
                post_relu_f_scaled,
            )
        else:
            return f

    def encode_decoupled(
        self,
        x: th.Tensor,
        return_active: bool = False,
        use_threshold: bool = True,
        select_features: list[int] | None = None,
    ):
        if select_features is not None and not use_threshold:
            raise ValueError(
                "select_features is not supported when use_threshold is False"
            )
        num_latents = (
            self.dict_size if select_features is None else len(select_features)
        )
        batch_size = x.size(0)
        post_relu_f = super().encode(x, select_features=select_features)
        code_normalization = self.get_code_normalization(select_features)
        post_relu_f_scaled = post_relu_f.unsqueeze(1) * code_normalization.unsqueeze(0)
        assert post_relu_f_scaled.shape == (
            x.shape[0],
            self.num_layers,
            num_latents,
        )
        if use_threshold:
            mask = post_relu_f_scaled > self.threshold.unsqueeze(0).unsqueeze(2)
            f = post_relu_f.unsqueeze(1) * mask
            if return_active:
                f_scaled = post_relu_f_scaled * mask
        else:
            # Flatten and perform batch top-k
            flattened_acts_scaled = post_relu_f_scaled.transpose(0, 1).flatten(
                start_dim=1
            )  # (num_layers, batch_size * dict_size)
            topk = flattened_acts_scaled.topk(self.k * batch_size, sorted=False, dim=-1)
            topk_mask = th.zeros(
                (self.num_layers, batch_size * self.dict_size),
                dtype=th.bool,
                device=post_relu_f.device,
            )
            topk_mask[
                th.arange(self.num_layers, device=post_relu_f.device).unsqueeze(1),
                topk.indices,
            ] = True
            topk_mask = topk_mask.reshape(
                self.num_layers, batch_size, self.dict_size
            ).transpose(0, 1)
            f = post_relu_f.unsqueeze(1) * topk_mask
            if return_active:
                f_scaled = post_relu_f_scaled * topk_mask
        assert f.shape == (
            batch_size,
            self.num_layers,
            num_latents,
        )
        active = f.sum(0).sum(0) > 0
        assert active.shape == (num_latents,)
        post_relu_f_scaled = post_relu_f_scaled.sum(dim=1)
        assert (
            post_relu_f_scaled.shape
            == post_relu_f.shape
            == (
                batch_size,
                num_latents,
            )
        )
        if return_active:
            assert f_scaled.shape == f.shape
            return (
                f,
                f_scaled,
                active,
                post_relu_f,
                post_relu_f_scaled,
            )
        else:
            return f

    def get_activations(
        self, x: th.Tensor, use_threshold: bool = True, select_features=None, **kwargs
    ):
        _, f_scaled, *_ = self.encode(
            x,
            use_threshold=use_threshold,
            return_active=True,
            select_features=select_features,
            **kwargs,
        )
        if self.decoupled_code:
            f_scaled = f_scaled.sum(1)
        assert f_scaled.shape == (x.shape[0], self.dict_size)
        return f_scaled

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
        from_hub: bool = False,
        **kwargs,
    ):
        """
        Load a pretrained crosscoder from a file.
        """
        if from_hub:
            return super().from_pretrained(
                path, device=device, dtype=dtype, from_hub=True, **kwargs
            )

        state_dict = th.load(path, map_location="cpu", weights_only=True)
        if "encoder.weight" not in state_dict:
            warn(
                "crosscoder state dict was saved while torch.compiled was enabled. Fixing..."
            )
            state_dict = {k.split("_orig_mod.")[1]: v for k, v in state_dict.items()}
        num_layers, activation_dim, dict_size = state_dict["encoder.weight"].shape
        if "code_normalization" in kwargs:
            code_normalization = kwargs["code_normalization"]
            kwargs.pop("code_normalization")
        elif "code_normalization_id" in state_dict:
            code_normalization = CodeNormalization._value2member_map_[
                state_dict["code_normalization_id"].item()
            ]
        elif "code_normalization" not in kwargs:
            warn(
                f"No code normalization id found in {path}. This is likely due to saving the model using an older version of dictionary_learning. Assuming code_normalization is CROSSCODER, if not pass code_normalization as a from_pretrained kwarg"
            )
            code_normalization = CodeNormalization.CROSSCODER
        if "k" in kwargs:
            assert (
                state_dict["k"] == kwargs["k"]
            ), f"k in kwargs ({kwargs['k']}) does not match k in state_dict ({state_dict['k']})"
            kwargs.pop("k")
        kwargs.update()
        crosscoder = cls(
            activation_dim,
            dict_size,
            num_layers,
            k=state_dict["k"],
            code_normalization=code_normalization,
            **kwargs,
        )
        if "code_normalization_id" not in state_dict:
            state_dict["code_normalization_id"] = th.tensor(
                code_normalization.value, dtype=th.int
            )
        crosscoder.load_state_dict(state_dict)

        if device is not None:
            crosscoder = crosscoder.to(device)
        return crosscoder.to(dtype=dtype)
