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
from abc import ABC, abstractmethod
from .utils import set_decoder_norm_to_unit_norm


class NormalizableMixin(nn.Module):
    """
    Mixin class providing activation normalization functionality.

    This mixin allows classes to optionally normalize and denormalize activations
    using mean and std tensors. If no mean/std is provided, activations
    pass through unchanged.
    """

    def __init__(
        self,
        activation_mean: th.Tensor | None = None,
        activation_std: th.Tensor | None = None,
        activation_shape: tuple[int, ...] | None = None,
        *,
        keep_relative_variance: bool = True,
        target_rms: float | None = 1.0,
    ):
        """
        Initialize the normalization mixin.

        Args:
            activation_mean: Optional mean tensor for normalization. If None,
                           normalization is a no-op.
            activation_std: Optional std tensor for normalization. If None,
                          normalization is a no-op.
            activation_shape: Shape of the activation tensor. Required if activation_mean and activation_std are None for proper initialization and registration of the buffers.
            keep_relative_variance: If True, performs global scaling so that the
                                  sum of variances is 1 while their relative magnitudes stay unchanged. If false we normalize neuron-wise. We normalize the last dimension.
            target_rms: Target RMS for input activation normalization.
        """
        super().__init__()
        self.keep_relative_variance = keep_relative_variance
        if target_rms is None:
            target_rms = 1.0
        self.register_buffer("target_rms", th.tensor(target_rms))
        if activation_mean is not None and activation_std is not None:
            # Type assertion to help linter understand these are tensors
            assert isinstance(
                activation_mean, th.Tensor
            ), "Expected mean to be a tensor"
            assert isinstance(activation_std, th.Tensor), "Expected std to be a tensor"
            assert not th.isnan(activation_mean).any(), "Expected mean to be non-NaN"
            assert not th.isnan(activation_std).any(), "Expected std to be non-NaN"
            self.register_buffer("activation_mean", activation_mean)
            self.register_buffer("activation_std", activation_std)
        else:
            assert (
                activation_shape is not None
            ), "activation_shape must be provided if activation_mean and activation_std are None"
            self.register_buffer("activation_mean", th.nan * th.ones(activation_shape))
            self.register_buffer("activation_std", th.nan * th.ones(activation_shape))

        if self.keep_relative_variance and self.has_activation_normalizer:
            total_var = (self.activation_std**2).sum(dim=-1)
            assert total_var.shape == self.activation_mean.shape[:-1]
            activation_global_scale = self.target_rms / th.sqrt(total_var + 1e-8)
            self.register_buffer("activation_global_scale", activation_global_scale)
        else:
            self.register_buffer(
                "activation_global_scale", th.ones(activation_shape[:-1])
            )

    @property
    def has_activation_normalizer(self) -> bool:
        """Check if activation normalization is enabled."""
        return (
            not th.isnan(self.activation_mean).any()
            and not th.isnan(self.activation_std).any()
        )

    def normalize_activations(self, x: th.Tensor, inplace: bool = False) -> th.Tensor:
        """
        Normalize input activations using the configured mean and std.

        Args:
            x: Input tensor to normalize
            inplace: If True, modify the input tensor in place

        Returns:
            Normalized tensor (same as input if no normalizer configured)
        """
        if self.has_activation_normalizer:
            if not inplace:
                x = x.clone()
            assert (
                x.shape[1:-1] == self.activation_global_scale.shape
            ), "Normalization shape mismatch"
            x = x - self.activation_mean

            if self.keep_relative_variance:
                return x * self.activation_global_scale.unsqueeze(0).unsqueeze(-1)
            else:
                return x / (self.activation_std + 1e-8)
        return x

    def denormalize_activations(self, x: th.Tensor, inplace: bool = False) -> th.Tensor:
        """
        Denormalize input activations using the configured mean and std.

        Args:
            x: Input tensor to denormalize
            inplace: If True, modify the input tensor in place

        Returns:
            Denormalized tensor (same as input if no normalizer configured)
        """
        if self.has_activation_normalizer:
            if not inplace:
                x = x.clone()
            assert (
                x.shape[1:-1] == self.activation_global_scale.shape
            ), "Normalization shape mismatch"

            if self.keep_relative_variance:
                x = x / (self.activation_global_scale.unsqueeze(0).unsqueeze(-1) + 1e-8)
            else:
                x = x * (self.activation_std + 1e-8)

            return x + self.activation_mean
        return x


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

    def save_pretrained(self, *args, **kwargs):
        """Save model with dict_class in config for local loading."""
        self._hub_mixin_config["dict_class"] = type(self).__name__
        return super().save_pretrained(*args, **kwargs)

    @classmethod
    def _from_pretrained(cls, *, model_id, **kwargs):
        """Strip dict_class from config before instantiation."""
        kwargs.pop("dict_class", None)
        return super()._from_pretrained(model_id=model_id, **kwargs)

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


class BatchTopKSAE(NormalizableMixin, Dictionary):
    """
    Batch Top-K Sparse Autoencoder implementation.

    This SAE uses a batch-wise top-k sparsity mechanism where only the top k features
    across the entire batch are kept active. This enforces global sparsity constraints
    rather than per-sample sparsity.

    Attributes:
        activation_dim: Dimension of input activations
        dict_size: Number of dictionary features
        k: Number of top features to keep active across the batch
        threshold: Threshold value for feature activation (when using threshold mode)
        encoder: Linear layer for encoding activations to features
        decoder: Linear layer for decoding features back to activations
        b_dec: Decoder bias parameter
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        activation_mean: th.Tensor | None = None,
        activation_std: th.Tensor | None = None,
        target_rms: float = 1.0,
        encoder_init_norm: float = 1.0,
    ):
        """
        Initialize the Batch Top-K SAE.

        Args:
            activation_dim: Dimension of the input activation vectors
            dict_size: Number of features in the dictionary
            k: Number of top features to keep active across the batch
            activation_mean: Optional mean tensor for input activation normalization. If None, no normalization is applied.
            activation_std: Optional std tensor for input activation normalization. If None, no normalization is applied.
            target_rms: Target variance for input activation normalization.
            encoder_init_norm: Norm for the encoder weights.
        """

        super().__init__(
            activation_mean=activation_mean,
            activation_std=activation_std,
            activation_shape=(activation_dim,),
            target_rms=target_rms,
        )

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
        self.encoder.weight.data = self.decoder.weight.T.clone() * encoder_init_norm
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(th.zeros(activation_dim))

    def encode(
        self,
        x: th.Tensor,
        return_active: bool = False,
        use_threshold: bool = True,
        normalize_activations: bool = True,
        inplace_normalize: bool = False,
    ):
        """
        Encode input activations to sparse feature representations.

        Args:
            x: Input activations of shape (batch_size, activation_dim)
            return_active: If True, return additional information about active features
            use_threshold: If True, use threshold-based sparsity; if False, use top-k
            normalize_activations: Whether to normalize input activations
            inplace_normalize: Whether to normalize activations in-place

        Returns:
            If return_active is False:
                encoded_acts_BF: Sparse feature activations (batch_size, dict_size)
            If return_active is True:
                Tuple of (encoded_acts_BF, active_features, post_relu_feat_acts_BF)
                where active_features indicates which features are active across the batch
        """
        if normalize_activations:
            x = self.normalize_activations(x, inplace=inplace_normalize)
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

    def decode(self, x: th.Tensor, denormalize_activations: bool = True) -> th.Tensor:
        """
        Decode sparse feature representations back to activations.

        Args:
            x: Sparse feature activations of shape (batch_size, dict_size)
            denormalize_activations: Whether to denormalize the output activations

        Returns:
            Reconstructed activations of shape (batch_size, activation_dim)
        """
        out = self.decoder(x) + self.b_dec
        if denormalize_activations:
            out = self.denormalize_activations(out, inplace=True)
        return out

    def forward(
        self,
        x: th.Tensor,
        output_features: bool = False,
        normalize_activations: bool = True,
    ):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input activations of shape (batch_size, activation_dim)
            output_features: If True, return both reconstructions and features
            normalize_activations: Whether to normalize input activations

        Returns:
            If output_features is False:
                x_hat_BD: Reconstructed activations (batch_size, activation_dim)
            If output_features is True:
                Tuple of (x_hat_BD, encoded_acts_BF) where encoded_acts_BF
                are the sparse feature activations
        """
        encoded_acts_BF = self.encode(x, normalize_activations=normalize_activations)
        x_hat_BD = self.decode(
            encoded_acts_BF, denormalize_activations=normalize_activations
        )

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        """
        Scale the bias parameters by a given factor.

        This is useful for adjusting the magnitude of biases during training
        or when changing the scale of input activations.

        Args:
            scale: Factor to multiply biases by
        """
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(
        cls, path, k=None, device=None, from_hub=False, **kwargs
    ) -> "BatchTopKSAE":
        """
        Load a pretrained BatchTopKSAE from a file or hub.

        Args:
            path: Path to the saved model file or hub identifier
            k: Number of top features to keep active. If None, use value from saved model
            device: Device to load the model on
            from_hub: Whether to load from the Hugging Face hub
            **kwargs: Additional arguments passed to hub loading

        Returns:
            Loaded BatchTopKSAE instance

        Raises:
            ValueError: If provided k doesn't match the saved model's k value
        """
        if from_hub:
            return super().from_pretrained(path, device=device, **kwargs)

        state_dict = th.load(path, weights_only=True)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        normalization_keys = [
            "target_rms",
            "activation_mean",
            "activation_std",
            "activation_global_scale",
        ]
        is_in_dict = th.tensor([k in state_dict for k in normalization_keys])
        if not is_in_dict.all():
            if is_in_dict.any():
                raise ValueError(
                    f"Some normalization keys are present in the state dict but not all. Missing keys: {[n for n in normalization_keys if n not in state_dict]}"
                )
            else:
                warn(
                    "No normalization keys found in the state dict. Assuming no normalization is needed. This is normal for old dictionaries."
                )
                for key in normalization_keys:
                    state_dict[key] = (
                        th.full((activation_dim,), th.nan)
                        if key in ["activation_mean", "activation_std"]
                        else th.tensor(th.nan)
                    )

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(
            activation_dim,
            dict_size,
            k,
        )
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    @property
    def device(self):
        return self.encoder.weight.device


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
    A crosscoder encoder that transforms multi-layer activations to dictionary features.

    This encoder processes activations from multiple layers simultaneously, applying
    layer-specific transformations and summing the results to produce a single
    dictionary feature representation.

    Attributes:
        activation_dim: Dimension of input activations for each layer
        dict_size: Number of features in the dictionary
        num_layers: Number of layers being encoded
        encoder_layers: List of layer indices to encode from
        weight: Learnable weight tensor of shape (num_layers, activation_dim, dict_size)
        bias: Learnable bias tensor of shape (dict_size,)
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
        """
        Initialize the CrossCoder encoder.

        Args:
            activation_dim: Dimension of the input activation vectors for each layer
            dict_size: Number of features in the dictionary
            num_layers: Total number of layers (required if encoder_layers not provided)
            same_init_for_all_layers: If True, initialize all layers with the same weights
            norm_init_scale: Scale factor for weight normalization after initialization
            encoder_layers: Specific layer indices to encode from (defaults to all layers)

        Raises:
            ValueError: If neither encoder_layers nor num_layers is specified
        """
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
        Convert multi-layer activations to dictionary features.

        Applies layer-specific linear transformations to each layer's activations,
        sums across layers, adds bias, and applies ReLU activation.

        Args:
            x: Input activations of shape (batch_size, n_layers, activation_dim)
            return_no_sum: If True, return both summed and per-layer features
            select_features: Optional list of feature indices to compute (for efficiency)
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            If return_no_sum is False:
                f: Dictionary features of shape (batch_size, dict_size)
            If return_no_sum is True:
                Tuple of (summed_features, per_layer_features) where:
                - summed_features: shape (batch_size, dict_size)
                - per_layer_features: shape (batch_size, num_layers, dict_size)
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
    A crosscoder decoder that transforms dictionary features back to multi-layer activations.

    This decoder reconstructs activations for multiple layers from a shared dictionary
    feature representation, applying layer-specific linear transformations.

    Attributes:
        activation_dim: Dimension of output activations for each layer
        dict_size: Number of features in the dictionary
        num_layers: Number of layers being decoded to
        weight: Learnable weight tensor of shape (num_layers, dict_size, activation_dim)
        bias: Learnable bias tensor of shape (num_layers, activation_dim)
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
        """
        Initialize the CrossCoder decoder.

        Args:
            activation_dim: Dimension of the output activation vectors for each layer
            dict_size: Number of features in the dictionary
            num_layers: Number of layers to decode to
            same_init_for_all_layers: If True, initialize all layers with the same weights
            norm_init_scale: Scale factor for weight normalization after initialization
            init_with_weight: Pre-initialized weight tensor to use instead of random init
        """
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
        Convert dictionary features back to multi-layer activations.

        Applies layer-specific linear transformations to convert dictionary features
        into activation representations for each layer.

        Args:
            f: Dictionary features of shape (batch_size, dict_size) or
               (batch_size, n_layers, dict_size)
            select_features: Optional list of feature indices to use (for efficiency)
            add_bias: Whether to add the decoder bias terms

        Returns:
            x: Reconstructed activations of shape (batch_size, n_layers, activation_dim)
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
    Enumeration of supported code normalization methods for dictionary learning.

    Code normalization determines how feature activations are scaled based on the
    decoder weights, affecting the sparsity penalty and feature interpretation.

    Values:
        CROSSCODER: Sum of norms of decoder weights across layers for each feature.
                   Encourages features to be active across multiple layers.
        SAE: Norm of concatenated decoder weights for each feature (as in standard SAEs).
             Treats all layers equally without cross-layer preference.
        MIXED: Weighted combination of SAE and CROSSCODER normalization.
               Allows balancing between single-layer and multi-layer feature activation.
        NONE: No normalization applied (uniform scaling of 1.0).
              Raw feature activations without decoder-weight-based scaling.
        DECOUPLED: Per-layer norms without summing across layers.
                  Each layer maintains separate feature scaling.
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
        String representation of the CodeNormalization.

        Returns:
            The name of the normalization type in uppercase
        """
        return self.name

    def __repr__(self) -> str:
        """
        String representation of the CodeNormalization.

        Returns:
            The name of the normalization type in uppercase
        """
        return self.name


class CrossCoder(Dictionary, NormalizableMixin):
    """
    A crosscoder sparse autoencoder for multi-layer activation processing.

    CrossCoders process activations from multiple layers simultaneously, learning
    a shared dictionary of features that can reconstruct activations across all layers.
    This enables discovery of features that span multiple computational layers.

    The architecture consists of:
    - Encoder: Maps multi-layer activations to dictionary features
    - Decoder: Reconstructs multi-layer activations from dictionary features
    - Optional latent processor: Transforms features between encoding and decoding

    Args:
        activation_dim: Dimension of input activations for each layer
        dict_size: Number of features in the dictionary
        num_layers: Number of layers to process
        same_init_for_all_layers: If True, initialize all layers with identical weights
        norm_init_scale: Scale factor for weight normalization (default: None)
        init_with_transpose: If True, initialize decoder as transpose of encoder
        encoder_layers: Specific layer indices to encode (default: all layers)
        latent_processor: Optional function to process features between encode/decode
        num_decoder_layers: Number of decoder layers (default: same as num_layers)
        code_normalization: Method for normalizing feature activations
        code_normalization_alpha_sae: Weight for SAE component in MIXED normalization
        code_normalization_alpha_cc: Weight for CrossCoder component in MIXED normalization
        activation_mean: Optional mean tensor for input/output activation normalization
        activation_std: Optional std tensor for input/output activation normalization
        target_rms: Optional target RMS for input/output activation normalization
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers=False,
        norm_init_scale: float | None = None,  # neel's default: 0.005
        init_with_transpose=True,
        encoder_layers: list[int] | None = None,
        latent_processor: Callable | None = None,
        num_decoder_layers: int | None = None,
        code_normalization: CodeNormalization | str = CodeNormalization.CROSSCODER,
        code_normalization_alpha_sae: float | None = 1.0,
        code_normalization_alpha_cc: float | None = 0.1,
        activation_mean: th.Tensor | None = None,
        activation_std: th.Tensor | None = None,
        target_rms: float | None = None,
    ):
        """
        Initialize a CrossCoder sparse autoencoder.

        Args:
            activation_dim: Dimension of input activations for each layer
            dict_size: Number of features in the dictionary
            num_layers: Number of layers to process
            same_init_for_all_layers: If True, initialize all layers with identical weights
            norm_init_scale: Scale factor for weight normalization after initialization
            init_with_transpose: If True, initialize decoder weights as encoder transpose
            encoder_layers: Specific layer indices to encode from (default: all layers)
            latent_processor: Optional function to process features between encode/decode
            num_decoder_layers: Number of decoder layers (default: same as num_layers)
            code_normalization: Method for normalizing feature activations
            code_normalization_alpha_sae: Weight for SAE component in MIXED normalization
            code_normalization_alpha_cc: Weight for CrossCoder component in MIXED normalization
            activation_mean: Optional mean tensor for input/output activation normalization
            activation_std: Optional std tensor for input/output activation normalization
            target_rms: Optional target RMS for input/output activation normalization
        """
        # First initialize the base classes that don't take normalization parameters
        super().__init__(
            activation_mean=activation_mean,
            activation_std=activation_std,
            activation_shape=(num_layers, activation_dim),
            target_rms=target_rms,
        )

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
        """
        Compute normalization weights for dictionary features based on decoder weights.

        Args:
            select_features: Optional list of feature indices to compute norms for.
                           If None, computes norms for all features.

        Returns:
            Normalization weights tensor with shape depending on normalization type:
            - SAE/MIXED: (1, dict_size)
            - CROSSCODER: (1, dict_size)
            - DECOUPLED: (n_layers, dict_size)
            - NONE: scalar tensor
        """
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
        self,
        x: th.Tensor,
        normalize_activations: bool = True,
        inplace_normalize: bool = False,
        **kwargs,
    ) -> th.Tensor:  # (batch_size, n_layers, dict_size)
        """
        Encode input activations to dictionary feature space.

        Args:
            x: Input activations of shape (batch_size, n_layers, activation_dim)
            normalize_activations: Whether to apply activation normalization before encoding
            inplace_normalize: Whether to normalize activations in-place
            **kwargs: Additional arguments passed to the encoder

        Returns:
            Encoded features of shape (batch_size, dict_size)
        """
        if normalize_activations:
            x = self.normalize_activations(x, inplace=inplace_normalize)
        return self.encoder(x, **kwargs)

    def get_activations(
        self,
        x: th.Tensor,
        use_threshold: bool = True,
        select_features=None,
        normalize_activations: bool = True,
        **kwargs,
    ):
        """
        Get normalized dictionary activations for input.

        Args:
            x: Input activations of shape (batch_size, n_layers, activation_dim)
            use_threshold: Whether to apply thresholding in encoder
            select_features: Optional list of feature indices to return. If None, returns all features.
            normalize_activations: Whether to apply activation normalization before encoding
            **kwargs: Additional arguments passed to encode

        Returns:
            Normalized activations scaled by code normalization weights.
            Shape: (batch_size, dict_size) or (batch_size, len(select_features))
        """
        f = self.encode(
            x,
            use_threshold=use_threshold,
            normalize_activations=normalize_activations,
            **kwargs,
        )
        weight_norm = self.get_code_normalization()
        if self.decoupled_code:
            weight_norm = weight_norm.sum(dim=0, keepdim=True)
        if select_features is not None:
            return (f * weight_norm)[:, select_features]
        return f * weight_norm

    def decode(
        self, f: th.Tensor, denormalize_activations: bool = True, **kwargs
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        """
        Decode dictionary features back to activation space.

        Args:
            f: Dictionary features of shape (batch_size, dict_size)
            denormalize_activations: Whether to denormalize output activations
            **kwargs: Additional arguments passed to the decoder

        Returns:
            Reconstructed activations of shape (batch_size, n_layers, activation_dim)
        """
        # f: (batch_size, n_layers, dict_size)
        x = self.decoder(f, **kwargs)
        if denormalize_activations:
            x = self.denormalize_activations(x, inplace=True)
        return x

    def forward(
        self, x: th.Tensor, output_features=False, normalize_activations: bool = True
    ):
        """
        Forward pass of the crosscoder.

        Args:
            x: Input activations of shape (batch_size, n_layers, activation_dim)
            output_features: If True, return both reconstructed activations and normalized features
            normalize_activations: Whether to apply activation normalization before encoding and denormalization after decoding

        Returns:
            If output_features=False:
                Reconstructed activations of shape (batch_size, n_layers, activation_dim)
            If output_features=True:
                Tuple of (reconstructed_activations, normalized_features) where:
                - reconstructed_activations: shape (batch_size, n_layers, activation_dim)
                - normalized_features: shape (batch_size, dict_size)
        """
        f = self.encode(x, normalize_activations=normalize_activations)
        if self.latent_processor is not None:
            f = self.latent_processor(f)
        x_hat = self.decode(f, denormalize_activations=normalize_activations)

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
        Load a pretrained crosscoder from a file or hub.

        Args:
            path: Path to model file or hub model identifier
            dtype: Data type to load the model with
            device: Device to load the model to
            from_hub: Whether to load from HuggingFace hub
            code_normalization: Override code normalization type if not found in state dict
            **kwargs: Additional arguments including activation_normalizer

        Returns:
            Loaded CrossCoder instance

        Raises:
            Warning: If no code normalization ID found in saved model
        """
        if isinstance(code_normalization, str):
            code_normalization = CodeNormalization.from_string(code_normalization)
        if from_hub or path.endswith(".safetensors"):
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

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    @property
    def device(self):
        return self.encoder.weight.device

    def resample_neurons(self, deads, activations):
        """
        Resample dead neurons by reinitializing their weights.

        Uses the resampling strategy from "Towards Monosemanticity" where dead neurons
        are reinitialized based on high-loss input examples. This helps recover neurons
        that have stopped activating during training.

        Args:
            deads: Boolean tensor of shape (dict_size,) indicating dead neurons
            activations: Input activations of shape (batch_size, num_layers, activation_dim)
                        used for resampling
        """
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
    """
    A CrossCoder variant that uses BatchTopK sparsity for feature selection.

    This implementation selects the top-k most active features across the entire batch,
    rather than applying a fixed threshold. This ensures a consistent level of sparsity
    and can help with training stability.

    Key features:
    - Adaptive thresholding based on activation statistics
    - Auxiliary loss for dead feature resurrection
    - Support for k-annealing (gradually reducing k during training)
    - Decoupled mode for per-layer thresholding

    Args:
        activation_dim: Dimension of input activations for each layer
        dict_size: Number of features in the dictionary
        num_layers: Number of layers to process
        k: Number of top features to keep active (can be int or tensor for annealing)
        norm_init_scale: Scale factor for weight initialization normalization
        activation_mean: Optional mean tensor for input/output activation normalization
        activation_std: Optional std tensor for input/output activation normalization
        *args: Additional positional arguments passed to parent CrossCoder
        **kwargs: Additional keyword arguments passed to parent CrossCoder
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        num_layers: int,
        k: int | th.Tensor = 100,
        norm_init_scale: float = 1.0,
        activation_mean: th.Tensor | None = None,
        activation_std: th.Tensor | None = None,
        target_rms: float | None = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize a BatchTopK CrossCoder.

        Args:
            activation_dim: Dimension of the input activations
            dict_size: Size of the dictionary/number of features
            num_layers: Number of layers in the crosscoder
            k: Number of top features to keep active. Can be int or tensor for k-annealing
            norm_init_scale: Scale factor for weight initialization normalization
            activation_mean: Optional mean tensor for input/output activation normalization
            activation_std: Optional std tensor for input/output activation normalization
            target_rms: Optional target RMS for input/output activation normalization
            *args: Additional positional arguments passed to parent class
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(
            activation_dim,
            dict_size,
            num_layers,
            norm_init_scale=norm_init_scale,
            activation_mean=activation_mean,
            activation_std=activation_std,
            target_rms=target_rms,
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
        normalize_activations: bool = True,
        inplace_normalize: bool = False,
    ):
        """
        Encode input activations using BatchTopK sparsity.

        This method applies either learned thresholding or top-k selection to enforce
        sparsity in the feature activations. In top-k mode, exactly k*batch_size features
        are kept active across the entire batch.

        Args:
            x: Input tensor of shape (batch_size, num_layers, activation_dim)
            return_active: If True, return additional activation information
            use_threshold: If True, use learned threshold; if False, use top-k selection
            select_features: Optional list of feature indices to select
            normalize_activations: Whether to normalize input activations
            inplace_normalize: Whether to normalize activations in-place

        Returns:
            If return_active is False:
                Encoded features tensor of shape (batch_size, dict_size) or
                (batch_size, num_layers, dict_size) for decoupled mode
            If return_active is True:
                Tuple of (features, scaled_features, active_mask, post_relu_features, post_relu_scaled_features)
        """
        if normalize_activations:
            x = self.normalize_activations(x, inplace=inplace_normalize)
        if self.decoupled_code:
            return self.encode_decoupled(
                x, return_active, use_threshold, select_features
            )
        batch_size = x.size(0)
        post_relu_f = super().encode(
            x, select_features=select_features, normalize_activations=False
        )
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
        """
        Encode input activations using decoupled BatchTopK sparsity.

        In decoupled mode, each layer maintains separate thresholds and top-k selection,
        allowing for layer-specific sparsity patterns while still sharing features across layers.

        Args:
            x: Input tensor of shape (batch_size, num_layers, activation_dim)
            return_active: If True, return additional activation information
            use_threshold: If True, use learned thresholds; if False, use top-k selection
            select_features: Optional list of feature indices to select

        Returns:
            If return_active is False:
                Encoded features tensor of shape (batch_size, num_layers, dict_size)
            If return_active is True:
                Tuple of (features, scaled_features, active_mask, post_relu_features, post_relu_scaled_features)

        Raises:
            ValueError: If select_features is used with use_threshold=False
        """
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
        self,
        x: th.Tensor,
        use_threshold: bool = True,
        select_features=None,
        normalize_activations: bool = True,
        inplace_normalize: bool = False,
        **kwargs,
    ):
        """
        Get scaled feature activations for the input.

        Args:
            x: Input tensor of shape (batch_size, num_layers, activation_dim)
            use_threshold: Whether to use learned threshold for sparsity
            select_features: Optional list of feature indices to select
            normalize_activations: Whether to normalize input activations
            inplace_normalize: Whether to normalize activations in-place
            **kwargs: Additional arguments passed to encode method

        Returns:
            Scaled feature activations tensor of shape (batch_size, num_features)
        """
        _, f_scaled, *_ = self.encode(
            x,
            use_threshold=use_threshold,
            return_active=True,
            select_features=select_features,
            normalize_activations=normalize_activations,
            inplace_normalize=inplace_normalize,
            **kwargs,
        )
        if self.decoupled_code:
            f_scaled = f_scaled.sum(1)
        assert f_scaled.shape == (
            x.shape[0],
            len(select_features) if select_features is not None else self.dict_size,
        )
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
        Load a pretrained BatchTopK CrossCoder from a file.

        Args:
            path: Path to the saved model file
            dtype: Target dtype for the loaded model
            device: Target device for the loaded model
            from_hub: Whether to load from a model hub
            **kwargs: Additional keyword arguments for model initialization

        Returns:
            Loaded BatchTopKCrossCoder instance

        Raises:
            AssertionError: If k in kwargs doesn't match k in saved state dict
            Warning: If no code normalization found in saved model
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

        crosscoder = cls(
            activation_dim,
            dict_size,
            num_layers,
            k=state_dict["k"],
            code_normalization=code_normalization,
            **kwargs,
        )

        crosscoder.load_state_dict(state_dict)
        if device is not None:
            crosscoder = crosscoder.to(device)
        return crosscoder.to(dtype=dtype)
