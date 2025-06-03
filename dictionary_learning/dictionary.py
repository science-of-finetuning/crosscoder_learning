# dictionary_learning/dictionary.py
import torch as th
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu
import einops
from warnings import warn
from typing import Callable, Optional, Union, List
from enum import Enum, auto
from abc import ABC, abstractmethod
from huggingface_hub import PyTorchModelHubMixin

from .utils import set_decoder_norm_to_unit_norm

# Base Class
class Dictionary(ABC, nn.Module, PyTorchModelHubMixin):
    dict_size: int
    activation_dim: int

    @abstractmethod
    def encode(self, x): pass
    @abstractmethod
    def decode(self, f): pass

    @classmethod
    @abstractmethod
    def from_pretrained( cls, path_or_model_id, from_hub=False, device=None, dtype=None, config=None, **kwargs ) -> "Dictionary":
        if from_hub:
            model = super(Dictionary, cls).from_pretrained(path_or_model_id, config=config, **kwargs) # type: ignore
        else:
            raise NotImplementedError("Subclasses must implement specific local from_pretrained if not using Hugging Face Hub.")

        if device is not None: model.to(device)
        if dtype is not None: model.to(dtype=dtype)
        return model # type: ignore

# Basic AutoEncoder (as it was in the existing file)
class AutoEncoder(Dictionary, nn.Module):
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self._hub_mixin_config = {"activation_dim": activation_dim, "dict_size": dict_size}
        self.bias = nn.Parameter(th.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = th.randn_like(self.decoder.weight)
        dec_weight = dec_weight / (dec_weight.norm(dim=0, keepdim=True) + 1e-8)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x): return nn.ReLU()(self.encoder(x - self.bias))
    def decode(self, f): return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        if ghost_mask is None:
            f = self.encode(x)
            x_hat = self.decode(f)
            return (x_hat, f) if output_features else x_hat
        else:
            f_pre = self.encoder(x-self.bias)
            f_ghost = th.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)
            x_ghost = self.decoder(f_ghost)
            x_hat = self.decode(f)
            return (x_hat, x_ghost, f) if output_features else (x_hat, x_ghost)

    @classmethod
    def from_pretrained(cls, path_or_model_id, dtype=th.float32, from_hub=False, device=None, config=None, **kwargs ) -> "AutoEncoder":
        cfg = config if config is not None else kwargs.get("config", {})
        act_dim = kwargs.get("activation_dim", cfg.get("activation_dim"))
        dct_size = kwargs.get("dict_size", cfg.get("dict_size"))

        if not all([act_dim, dct_size]):
            if not from_hub and path_or_model_id:
                try:
                    sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                    dec_w_shape = sd.get('decoder.weight.shape', sd.get('decoder.weight').shape)
                    act_dim_sd, dct_size_sd = dec_w_shape[0], dec_w_shape[1]
                    if act_dim is None: act_dim = act_dim_sd
                    if dct_size is None: dct_size = dct_size_sd
                except Exception as e:
                    raise ValueError(f"Could not load/infer params from local state_dict {path_or_model_id} for AutoEncoder: {e}")
            if not all([act_dim, dct_size]):
                raise ValueError("activation_dim, dict_size must be provided in config or kwargs for AutoEncoder.")

        cfg.update({"activation_dim": act_dim, "dict_size": dct_size})

        if from_hub:
            model = super(AutoEncoder, cls).from_pretrained(path_or_model_id, config=cfg, **kwargs) # type: ignore
        else:
            instance = cls(activation_dim=act_dim, dict_size=dct_size)
            state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
            if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
            instance.load_state_dict(state_dict)
            model = instance

        model.to(dtype=dtype) # type: ignore
        if device: model.to(device) # type: ignore
        return model # type: ignore

class IdentityDict(Dictionary, nn.Module):
    def __init__(self, activation_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.activation_dim = activation_dim # type: ignore
        self.dict_size = activation_dim # type: ignore
        self._hub_mixin_config = {}
        if activation_dim is not None:
            self._hub_mixin_config["activation_dim"] = activation_dim

    def encode(self, x): return x
    def decode(self, f): return f
    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features: return x, x
        else: return x

    @classmethod
    def from_pretrained(cls, path_or_model_id, dtype=th.float32, device=None, from_hub=False, config=None, **kwargs):
        cfg = config if config is not None else kwargs.get("config", {})
        act_dim = kwargs.get("activation_dim", cfg.get("activation_dim"))
        cfg.update({"activation_dim": act_dim})
        if from_hub:
            model = super(IdentityDict, cls).from_pretrained(path_or_model_id, config=cfg, **kwargs) # type: ignore
        else:
            model = cls(activation_dim=act_dim)
        model.to(dtype=dtype) # type: ignore
        if device: model.to(device) # type: ignore
        return model # type: ignore

class GatedAutoEncoder(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, initialization: Union[str, Callable]="default", device: Optional[Union[str, th.device]]=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self._hub_mixin_config = {"activation_dim": activation_dim, "dict_size": dict_size,
                                  "initialization_type": "default" if isinstance(initialization, str) else "custom"}
        param_device = device if device is not None else (th.empty(0).device)
        self.decoder_bias = nn.Parameter(th.empty(activation_dim, device=param_device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=param_device)
        self.r_mag = nn.Parameter(th.empty(dict_size, device=param_device))
        self.gate_bias = nn.Parameter(th.empty(dict_size, device=param_device))
        self.mag_bias = nn.Parameter(th.empty(dict_size, device=param_device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=param_device)
        if isinstance(initialization, str) and initialization == "default":
            self._reset_parameters()
        elif callable(initialization):
            initialization(self)

    def _reset_parameters(self):
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)
        dec_weight = th.randn_like(self.decoder.weight)
        dec_weight = dec_weight / (dec_weight.norm(dim=0, keepdim=True) + 1e-8)
        self.decoder.weight = nn.Parameter(dec_weight)
        init.kaiming_uniform_(self.encoder.weight, a=0.01)

    def encode(self, x, return_gate=False):
        x_enc = self.encoder(x - self.decoder_bias)
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)
        f = f_gate * f_mag
        f_scaled = f * (self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
        return (f_scaled, nn.ReLU()(pi_gate)) if return_gate else f_scaled

    def decode(self, f):
        f_unscaled = f / (self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
        return self.decoder(f_unscaled) + self.decoder_bias

    def forward(self, x, output_features=False):
        f_scaled = self.encode(x)
        x_hat = self.decode(f_scaled)
        return (x_hat, f_scaled) if output_features else x_hat

    @classmethod
    def from_pretrained(cls, path_or_model_id, from_hub=False, device=None, dtype=th.float32, config=None, **kwargs):
        cfg = config if config is not None else kwargs.get("config", {})
        act_dim = kwargs.get("activation_dim", cfg.get("activation_dim"))
        dct_size = kwargs.get("dict_size", cfg.get("dict_size"))
        initialization = cfg.get("initialization_type", "default")

        if not all([act_dim, dct_size]):
            if not from_hub and path_or_model_id:
                try:
                    sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                    act_dim_sd, dct_size_sd = sd["decoder.weight"].shape[0], sd["decoder.weight"].shape[1]
                    if act_dim is None: act_dim = act_dim_sd
                    if dct_size is None: dct_size = dct_size_sd
                except Exception as e:
                    raise ValueError(f"Could not load/infer params from local state_dict {path_or_model_id} for GatedAutoEncoder: {e}")
            if not all([act_dim, dct_size]):
                raise ValueError("activation_dim, dict_size required for GatedAutoEncoder.")
        cfg.update({"activation_dim":act_dim, "dict_size":dct_size, "initialization_type": initialization})
        if from_hub:
            model = super(GatedAutoEncoder, cls).from_pretrained(path_or_model_id, config=cfg, initialization=initialization, **kwargs) # type: ignore
        else:
            instance = cls(activation_dim=act_dim, dict_size=dct_size, initialization=initialization, device=device)
            state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
            if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
            instance.load_state_dict(state_dict)
            model = instance
        model.to(dtype=dtype) # type: ignore
        if device: model.to(device) # type: ignore
        return model # type: ignore

class JumpReluAutoEncoder(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, device: Optional[Union[str, th.device]]="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self._hub_mixin_config = {"activation_dim": activation_dim, "dict_size": dict_size}
        self.W_enc = nn.Parameter(th.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(th.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(th.empty(dict_size, activation_dim, device=device))
        self.b_dec = nn.Parameter(th.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(th.zeros(dict_size, device=device))
        self.apply_b_dec_to_input = False
        self.W_enc.data = th.randn_like(self.W_enc)
        self.W_enc.data = self.W_enc / (self.W_enc.norm(dim=0, keepdim=True) + 1e-8)
        self.W_dec.data = self.W_enc.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input: x = x - self.b_dec
        pre_linear = x @ self.W_enc
        pre_jump = pre_linear + self.b_enc
        f_raw = pre_jump * (pre_jump > self.threshold)
        f = nn.ReLU()(f_raw)
        f_scaled = f * (self.W_dec.norm(dim=1, keepdim=True).T + 1e-8)
        return (f_scaled, pre_jump) if output_pre_jump else f_scaled

    def decode(self, f):
        f_unscaled = f / (self.W_dec.norm(dim=1, keepdim=True).T + 1e-8)
        return f_unscaled @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        f_scaled = self.encode(x)
        x_hat = self.decode(f_scaled)
        return (x_hat, f_scaled) if output_features else x_hat

    @classmethod
    def from_pretrained(cls, path_or_model_id, from_hub=False, device=None, dtype=th.float32, config=None, load_from_sae_lens=False, **kwargs):
        cfg = config if config is not None else kwargs.get("config", {})
        act_dim = kwargs.get("activation_dim", cfg.get("activation_dim"))
        dct_size = kwargs.get("dict_size", cfg.get("dict_size"))
        apply_b_dec_to_input_cfg = cfg.get("apply_b_dec_to_input", False)
        if load_from_sae_lens:
            try: from sae_lens import SAE
            except ImportError: raise ImportError("sae_lens package not found. Please install it to load from SAE Lens checkpoints.")
            sae_kwargs = {k:v for k,v in kwargs.items() if k not in ['activation_dim', 'dict_size', 'config', 'load_from_sae_lens', 'apply_b_dec_to_input']}
            if device: sae_kwargs['device'] = str(device)
            sae, cfg_dict_sl, _ = SAE.from_pretrained(path_or_model_id, **sae_kwargs)
            act_dim = cfg_dict_sl["d_in"]
            dct_size = cfg_dict_sl["d_sae"]
            instance = cls(activation_dim=act_dim, dict_size=dct_size, device=sae.device)
            sae_state_dict = sae.state_dict()
            new_state_dict = {
                'W_enc': sae_state_dict['W_enc'].T, 'b_enc': sae_state_dict['b_enc'],
                'W_dec': sae_state_dict['W_dec'].T, 'b_dec': sae_state_dict['b_dec'],}
            if 'scaling_factor' in sae_state_dict:
                warn("SAE Lens 'scaling_factor' found but not used for 'threshold' in JumpReluAutoEncoder.")
            instance.load_state_dict(new_state_dict, strict=False)
            instance.apply_b_dec_to_input = cfg_dict_sl.get("apply_b_dec_to_input", apply_b_dec_to_input_cfg)
            model = instance
        else:
            if not all([act_dim, dct_size]):
                if not from_hub and path_or_model_id:
                    try:
                        sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                        act_dim_sd, dct_size_sd = sd["W_enc"].shape[0], sd["W_enc"].shape[1]
                        if act_dim is None: act_dim = act_dim_sd
                        if dct_size is None: dct_size = dct_size_sd
                    except Exception as e:
                         raise ValueError(f"Could not load/infer params from local state_dict {path_or_model_id} for JumpReluAutoEncoder: {e}")
                if not all([act_dim, dct_size]):
                    raise ValueError("activation_dim, dict_size required for JumpReluAutoEncoder.")
            cfg.update({"activation_dim":act_dim, "dict_size":dct_size, "apply_b_dec_to_input": apply_b_dec_to_input_cfg})
            if from_hub:
                model = super(JumpReluAutoEncoder, cls).from_pretrained(path_or_model_id, config=cfg, **kwargs) # type: ignore
            else:
                instance = cls(activation_dim=act_dim, dict_size=dct_size, device=device if device else "cpu")
                state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
                if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
                instance.load_state_dict(state_dict)
                model = instance
        model.to(dtype=dtype) # type: ignore
        if device: model.to(device) # type: ignore
        return model # type: ignore

class AutoEncoderNew(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self._hub_mixin_config = {"activation_dim": activation_dim, "dict_size": dict_size}
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)
        w_enc_init = th.randn(dict_size, activation_dim)
        w_enc_init = w_enc_init / (w_enc_init.norm(dim=1, keepdim=True) + 1e-8)
        self.encoder.weight = nn.Parameter(w_enc_init.clone())
        w_dec_init = th.randn(activation_dim, dict_size)
        w_dec_init = w_dec_init / (w_dec_init.norm(dim=0, keepdim=True) + 1e-8)
        self.decoder.weight = nn.Parameter(w_dec_init.clone())
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x: th.Tensor) -> th.Tensor:
        return nn.ReLU()(self.encoder(x))
    def decode(self, f: th.Tensor) -> th.Tensor:
        return self.decoder(f)
    def forward(self, x: th.Tensor, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            f_scaled = f * (self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
            return x_hat, f_scaled
        return x_hat
    @classmethod
    def from_pretrained(cls, path_or_model_id, from_hub=False, device=None, dtype=th.float32, config=None, **kwargs):
        cfg = config if config is not None else kwargs.get("config", {})
        act_dim = kwargs.get("activation_dim", cfg.get("activation_dim"))
        dct_size = kwargs.get("dict_size", cfg.get("dict_size"))
        if not all([act_dim, dct_size]):
            if not from_hub and path_or_model_id:
                try:
                    sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                    dct_size_sd, act_dim_sd = sd["encoder.weight"].shape
                    if act_dim is None: act_dim = act_dim_sd
                    if dct_size is None: dct_size = dct_size_sd
                except Exception as e:
                    raise ValueError(f"Could not load/infer params from local state_dict {path_or_model_id} for AutoEncoderNew: {e}")
            if not all([act_dim, dct_size]):
                raise ValueError("activation_dim, dict_size required for AutoEncoderNew.")
        cfg.update({"activation_dim":act_dim, "dict_size":dct_size})
        if from_hub:
            model = super(AutoEncoderNew, cls).from_pretrained(path_or_model_id, config=cfg, **kwargs) # type: ignore
        else:
            instance = cls(activation_dim=act_dim, dict_size=dct_size)
            state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
            if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
            instance.load_state_dict(state_dict)
            model = instance
        model.to(dtype=dtype); # type: ignore
        if device: model.to(device); # type: ignore
        return model # type: ignore

class BatchTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, decoder_type: str = "linear"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_type = decoder_type
        self._hub_mixin_config = {"activation_dim": activation_dim, "dict_size": dict_size, "k": k, "decoder_type": decoder_type}
        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", th.tensor(k, dtype=th.int))
        self.register_buffer("threshold", th.tensor(-1.0, dtype=th.float32))
        if self.decoder_type == "linear":
            self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
            self.decoder.weight.data = set_decoder_norm_to_unit_norm(self.decoder.weight, activation_dim, dict_size)
            self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
            self.encoder.weight.data = self.decoder.weight.T.clone()
        elif self.decoder_type == "embedding_bag":
            temp_decoder_linear = nn.Linear(dict_size, activation_dim, bias=False)
            temp_decoder_linear.weight.data = set_decoder_norm_to_unit_norm(temp_decoder_linear.weight, activation_dim, dict_size)
            self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
            self.encoder.weight.data = temp_decoder_linear.weight.T.clone()
            self.decoder = nn.EmbeddingBag(dict_size, activation_dim, mode="sum", sparse=True)
            self.decoder.weight = nn.Parameter(temp_decoder_linear.weight.T.clone())
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        if self.encoder.bias is not None: self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(th.zeros(activation_dim))

    def encode(self, x: th.Tensor, return_active: bool = False, use_threshold: bool = True):
        x_minus_bias = x - self.b_dec
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x_minus_bias))
        encoded_acts_BF: th.Tensor
        if use_threshold and self.threshold.item() >= 0:
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
        else:
            batch_size = x.size(0)
            k_actual = self.k.item()
            total_elements = post_relu_feat_acts_BF.numel()
            num_to_select = min(k_actual * batch_size, total_elements if total_elements > 0 else 1)
            if num_to_select <= 0 :
                 encoded_acts_BF = th.zeros_like(post_relu_feat_acts_BF)
            else:
                flattened_acts = post_relu_feat_acts_BF.flatten()
                post_topk = flattened_acts.topk(num_to_select, sorted=False, dim=-1)
                encoded_acts_BF = (
                    th.zeros_like(post_relu_feat_acts_BF.flatten())
                    .scatter_(-1, post_topk.indices, post_topk.values)
                    .reshape(post_relu_feat_acts_BF.shape))
        if return_active:
            active_indices_F = (encoded_acts_BF.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            return encoded_acts_BF, active_indices_F, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: th.Tensor) -> th.Tensor:
        if self.decoder_type == "embedding_bag":
            batch_size = x.size(0)
            all_indices, all_weights, offsets, any_active = [], [], [0], False
            current_offset = 0
            for i_sample in range(batch_size):
                sample_activations = x[i_sample]
                sample_indices = sample_activations.nonzero(as_tuple=True)[0]
                if sample_indices.numel() > 0:
                    any_active = True
                    all_indices.append(sample_indices)
                    all_weights.append(sample_activations[sample_indices])
                    current_offset += sample_indices.numel()
                offsets.append(current_offset)
            if not any_active:
                return self.b_dec.unsqueeze(0).expand(batch_size, -1) if self.b_dec is not None else \
                       th.zeros(batch_size, self.activation_dim, device=x.device, dtype=x.dtype)
            all_indices_cat = th.cat(all_indices)
            all_weights_cat = th.cat(all_weights)
            offsets_tensor = th.tensor(offsets[:-1], device=x.device, dtype=th.long)
            decoded_x = self.decoder(input=all_indices_cat, per_sample_weights=all_weights_cat, offsets=offsets_tensor)
            return decoded_x + self.b_dec
        else:
            return self.decoder(x) + self.b_dec

    def forward(self, x: th.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features: return x_hat_BD
        else: return x_hat_BD, encoded_acts_BF

    @classmethod
    def from_pretrained( cls, path_or_model_id, from_hub=False, device=None, dtype=th.float32, config=None, **kwargs ) -> "BatchTopKSAE":
        cfg_init = config if config is not None else kwargs.get("config", {})
        act_dim = kwargs.get("activation_dim", cfg_init.get("activation_dim"))
        dct_size = kwargs.get("dict_size", cfg_init.get("dict_size"))
        k_val = kwargs.get("k", cfg_init.get("k"))
        dec_type = kwargs.get("decoder_type", cfg_init.get("decoder_type", "linear"))
        if not all([act_dim, dct_size, k_val is not None]):
            if not from_hub and path_or_model_id:
                try:
                    sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                    if "_hub_mixin_config" in sd : cfg_init.update(sd["_hub_mixin_config"])
                    if act_dim is None: act_dim = cfg_init.get("activation_dim")
                    if dct_size is None: dct_size = cfg_init.get("dict_size")
                    if k_val is None: k_val = cfg_init.get("k")
                    if dec_type is None: dec_type = cfg_init.get("decoder_type", "linear")
                    if k_val is None and 'k' in sd : k_val = sd['k'].item()
                except Exception as e:
                    raise ValueError(f"Could not load/infer params from local state_dict {path_or_model_id} for BatchTopKSAE: {e}")
            if not all([act_dim, dct_size, k_val is not None]):
                 raise ValueError("activation_dim, dict_size, k must be provided for BatchTopKSAE.")
        cfg_init.update({"activation_dim": act_dim, "dict_size": dct_size, "k": k_val, "decoder_type": dec_type})
        if from_hub:
            model = super(BatchTopKSAE, cls).from_pretrained(path_or_model_id, config=cfg_init, **kwargs) # type: ignore
        else:
            instance = cls(activation_dim=act_dim, dict_size=dct_size, k=k_val, decoder_type=dec_type) # type: ignore
            state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
            if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
            instance.load_state_dict(state_dict)
            model = instance
        model.to(dtype=dtype); # type: ignore
        if device: model.to(device); # type: ignore
        return model # type: ignore

class CrossCoderEncoder(nn.Module):
    def __init__( self, activation_dim, dict_size, num_layers=None,
                  same_init_for_all_layers: bool = False, norm_init_scale: float | None = None,
                  encoder_layers_indices: list[int] | None = None ):
        super().__init__()
        if encoder_layers_indices is None:
            if num_layers is None: raise ValueError("num_layers or encoder_layers_indices must be given.")
            self.encoder_layers_indices = list(range(num_layers))
            self.num_encoder_matrices = num_layers
        else:
            self.encoder_layers_indices = encoder_layers_indices
            self.num_encoder_matrices = len(encoder_layers_indices)
        self.activation_dim = activation_dim; self.dict_size = dict_size
        if same_init_for_all_layers:
            w_single = init.kaiming_uniform_(th.empty(activation_dim, dict_size))
            if norm_init_scale is not None: w_single = w_single / (w_single.norm(dim=0, keepdim=True)+1e-8) * norm_init_scale
            self.weight = nn.Parameter(w_single.repeat(self.num_encoder_matrices, 1, 1))
        else:
            w_data = init.kaiming_uniform_(th.empty(self.num_encoder_matrices, activation_dim, dict_size))
            if norm_init_scale is not None: w_data = w_data / (w_data.norm(dim=1, keepdim=True)+1e-8) * norm_init_scale
            self.weight = nn.Parameter(w_data)
        self.bias = nn.Parameter(th.zeros(dict_size))

    def forward( self, x: th.Tensor, return_no_sum: bool = False, select_features: list[int] | None = None, **kwargs ) -> th.Tensor:
        x_selected = x[:, self.encoder_layers_indices]; w = self.weight; b = self.bias
        if select_features is not None: w = w[:, :, select_features]; b = b[select_features]
        f_per_layer = th.einsum("bld, ldf -> blf", x_selected, w); f_summed = f_per_layer.sum(dim=1)
        if not return_no_sum: return relu(f_summed + b)
        else: return relu(f_summed + b), relu(f_per_layer + b.view(1,1,-1))

class CrossCoderDecoder(nn.Module):
    def __init__( self, activation_dim: int, dict_size: int, num_output_layers: int,
                  decoder_type: str = "linear", same_init_for_all_layers: bool = False,
                  norm_init_scale: float | None = None, init_with_weight: th.Tensor | None = None ):
        super().__init__()
        self.activation_dim = activation_dim; self.dict_size = dict_size
        self.num_output_layers = num_output_layers; self.decoder_type = decoder_type
        self.bias = nn.Parameter(th.zeros(num_output_layers, activation_dim))
        if self.decoder_type == "linear":
            if init_with_weight is not None:
                if init_with_weight.shape != (num_output_layers, dict_size, activation_dim):
                    raise ValueError(f"init_with_weight shape {init_with_weight.shape} incompatible.")
                self.weight = nn.Parameter(init_with_weight.clone())
            else:
                if same_init_for_all_layers:
                    w_s = init.kaiming_uniform_(th.empty(dict_size, activation_dim))
                    if norm_init_scale: w_s = w_s / (w_s.norm(dim=1,keepdim=True)+1e-8)*norm_init_scale
                    self.weight = nn.Parameter(w_s.repeat(num_output_layers,1,1))
                else:
                    w_d = init.kaiming_uniform_(th.empty(num_output_layers,dict_size,activation_dim))
                    if norm_init_scale: w_d = w_d / (w_d.norm(dim=2,keepdim=True)+1e-8)*norm_init_scale
                    self.weight = nn.Parameter(w_d)
        elif self.decoder_type == "embedding_bag":
            self.layers = nn.ModuleList()
            for i in range(num_output_layers):
                eb_layer = nn.EmbeddingBag(dict_size, activation_dim, mode="sum", sparse=True)
                if init_with_weight is not None:
                    if init_with_weight.shape != (num_output_layers, dict_size, activation_dim):
                         raise ValueError(f"init_with_weight shape {init_with_weight.shape} incompatible for EB layers.")
                    eb_layer.weight = nn.Parameter(init_with_weight[i].clone())
                else:
                    temp_eb_w = init.kaiming_uniform_(th.empty(dict_size, activation_dim))
                    if norm_init_scale:
                        temp_eb_w = temp_eb_w / (temp_eb_w.norm(dim=0, keepdim=True)+1e-8) * norm_init_scale
                    eb_layer.weight = nn.Parameter(temp_eb_w)
                self.layers.append(eb_layer)
        else: raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward( self, f: th.Tensor, select_features: list[int] | None = None, add_bias: bool = True) -> th.Tensor:
        batch_size = f.size(0)
        if self.decoder_type == "embedding_bag":
            if select_features: raise NotImplementedError("select_features with EB in CrossCoderDecoder.")
            layer_outputs = []
            is_decoupled_f = f.dim() == 3
            if is_decoupled_f and f.shape[1] != self.num_output_layers:
                raise ValueError(f"Input f has {f.shape[1]} layers, EB decoder has {self.num_output_layers} layers.")
            for l_idx in range(self.num_output_layers):
                eb_layer = self.layers[l_idx]; f_for_layer = f[:,l_idx,:] if is_decoupled_f else f
                all_idx, all_w, offs, any_act = [], [], [0], False; curr_off = 0
                for i_s in range(batch_size):
                    s_act = f_for_layer[i_s]; s_idx = s_act.nonzero(as_tuple=True)[0]
                    if s_idx.numel() > 0: any_act=True; all_idx.append(s_idx); all_w.append(s_act[s_idx]); curr_off+=s_idx.numel()
                    offs.append(curr_off)
                if not any_act: layer_outputs.append(th.zeros(batch_size, self.activation_dim, device=f.device, dtype=f.dtype))
                else:
                    all_idx_c = th.cat(all_idx); all_w_c = th.cat(all_w)
                    offs_t = th.tensor(offs[:-1], device=f.device, dtype=th.long)
                    layer_outputs.append(eb_layer(input=all_idx_c, per_sample_weights=all_w_c, offsets=offs_t))
            x_stacked = th.stack(layer_outputs, dim=1)
            if add_bias: x_stacked += self.bias
            return x_stacked
        else:
            w = self.weight
            if select_features: w = w[:, select_features, :]
            if f.dim() == 2: x = th.einsum("bf, lfd -> bld", f, w)
            elif f.dim() == 3:
                if f.shape[1] != self.num_output_layers: raise ValueError(f"Input f has {f.shape[1]} layers, Linear decoder expects {self.num_output_layers}.")
                x = th.einsum("blf, lfd -> bld", f, w)
            else: raise ValueError(f"Unsupported f shape: {f.shape}")
            if add_bias: x += self.bias
            return x

class CodeNormalization(Enum):
    CROSSCODER = auto(); SAE = auto(); MIXED = auto(); NONE = auto(); DECOUPLED = auto()
    @classmethod
    def from_string(cls, s: str):
        try: return cls[s.upper()]
        except KeyError: raise ValueError(f"Unknown CodeNormalization: {s}")
    def __str__(self): return self.name; __repr__ = __str__

class CrossCoder(Dictionary, nn.Module):
    def __init__(
        self, activation_dim: int, dict_size: int, num_encoder_layers: int,
        decoder_type: str = "linear",
        same_init_for_all_layers: bool = False, norm_init_scale: float | None = None,
        init_with_transpose: bool = True, encoder_layers_indices: list[int] | None = None,
        latent_processor: Callable | None = None, num_decoder_output_layers: int | None = None,
        code_normalization: Union[CodeNormalization, str] = CodeNormalization.CROSSCODER,
        code_normalization_alpha_sae: float = 1.0,
        code_normalization_alpha_cc: float = 0.1,
    ):
        super().__init__()
        _num_enc_matrices = len(encoder_layers_indices) if encoder_layers_indices is not None else num_encoder_layers
        _actual_encoder_indices = encoder_layers_indices if encoder_layers_indices is not None else list(range(num_encoder_layers))
        _num_dec_out_layers = num_decoder_output_layers if num_decoder_output_layers is not None else _num_enc_matrices
        self.activation_dim = activation_dim; self.dict_size = dict_size
        self.num_encoder_layers_config = num_encoder_layers
        self.num_encoder_layers_processed = _num_enc_matrices
        self.latent_processor = latent_processor
        self.code_normalization = CodeNormalization.from_string(code_normalization) if isinstance(code_normalization, str) else code_normalization
        self.code_normalization_alpha_sae = code_normalization_alpha_sae
        self.code_normalization_alpha_cc = code_normalization_alpha_cc
        self._hub_mixin_config = {
            "activation_dim": activation_dim, "dict_size": dict_size,
            "num_encoder_layers": num_encoder_layers, "decoder_type": decoder_type,
            "same_init_for_all_layers":same_init_for_all_layers, "norm_init_scale":norm_init_scale,
            "init_with_transpose":init_with_transpose, "encoder_layers_indices":encoder_layers_indices,
            "num_decoder_output_layers":num_decoder_output_layers,
            "code_normalization": self.code_normalization.name,
            "code_normalization_alpha_sae": code_normalization_alpha_sae,
            "code_normalization_alpha_cc": code_normalization_alpha_cc,
        }
        self.encoder = CrossCoderEncoder(
            activation_dim, dict_size, num_layers=_num_enc_matrices,
            encoder_layers_indices=_actual_encoder_indices,
            same_init_for_all_layers=same_init_for_all_layers, norm_init_scale=norm_init_scale)
        dec_init_w = None
        if init_with_transpose:
            enc_w_for_transpose = self.encoder.weight.data
            if _num_enc_matrices != _num_dec_out_layers:
                warn(f"CrossCoder init_with_transpose: encoder matrices ({_num_enc_matrices}) != decoder output layers ({_num_dec_out_layers}).")
                if _num_enc_matrices > _num_dec_out_layers: enc_w_for_transpose = enc_w_for_transpose[:_num_dec_out_layers]
            if enc_w_for_transpose.shape[0] == _num_dec_out_layers : dec_init_w = einops.rearrange(enc_w_for_transpose, "l d f -> l f d")
            else: dec_init_w = None; warn("Cannot fully init decoder with transpose due to layer count mismatch.")
        self.decoder = CrossCoderDecoder(
            activation_dim, dict_size, _num_dec_out_layers, decoder_type=decoder_type,
            same_init_for_all_layers=same_init_for_all_layers, init_with_weight=dec_init_w,
            norm_init_scale=norm_init_scale)
        self.register_buffer("code_normalization_id", th.tensor(self.code_normalization.value))
        self.decoupled_code = self.code_normalization == CodeNormalization.DECOUPLED

    def get_code_normalization(self, select_features: list[int] | None = None) -> th.Tensor:
        dw: th.Tensor; fallback_device = self.encoder.weight.device if hasattr(self.encoder, 'weight') else th.empty(0).device
        if self.decoder.decoder_type == "embedding_bag":
            if not hasattr(self.decoder, 'layers') or not self.decoder.layers: return th.tensor(1.0, device=fallback_device)
            dw = th.stack([layer.weight.data for layer in self.decoder.layers], dim=0)
        else:
            if not hasattr(self.decoder, 'weight') or self.decoder.weight is None: return th.tensor(1.0, device=fallback_device)
            dw = self.decoder.weight
        if select_features: dw = dw[:, select_features, :]
        weight_norm: th.Tensor
        if self.code_normalization == CodeNormalization.SAE: weight_norm = dw.permute(1,0,2).flatten(1).norm(dim=1, keepdim=True).T
        elif self.code_normalization == CodeNormalization.MIXED:
            wn_sae = dw.permute(1,0,2).flatten(1).norm(dim=1, keepdim=True).T
            wn_cc = dw.norm(dim=2).sum(dim=0, keepdim=True)
            weight_norm = wn_sae * self.code_normalization_alpha_sae + wn_cc * self.code_normalization_alpha_cc
        elif self.code_normalization == CodeNormalization.NONE:
            num_feats = dw.shape[1]
            target_shape = (self.decoder.num_output_layers, num_feats) if self.decoupled_code else (1, num_feats)
            weight_norm = th.ones(target_shape, device=dw.device, dtype=dw.dtype)
        elif self.code_normalization == CodeNormalization.CROSSCODER: weight_norm = dw.norm(dim=2).sum(dim=0, keepdim=True)
        elif self.code_normalization == CodeNormalization.DECOUPLED: weight_norm = dw.norm(dim=2)
        else: raise NotImplementedError(f"Code norm {self.code_normalization} unknown.")
        return weight_norm

    def encode(self, x: th.Tensor, **kwargs) -> th.Tensor: return self.encoder(x, **kwargs)
    def decode(self, f: th.Tensor, **kwargs) -> th.Tensor: return self.decoder(f, **kwargs)
    def forward(self, x: th.Tensor, output_features=False):
        f = self.encode(x);
        if self.latent_processor: f = self.latent_processor(f)
        x_hat = self.decode(f)
        if output_features:
            norm = self.get_code_normalization()
            if f.dim()==2 and norm.dim()==2 and norm.shape[0] > 1 and norm.shape[0] == self.decoder.num_output_layers :
                norm = norm.sum(dim=0, keepdim=True)
            f_scaled = f * norm; return x_hat, f_scaled
        return x_hat

    @classmethod
    def from_pretrained(cls, path_or_model_id, dtype=th.float32, device=None, from_hub=False, config=None, **kwargs):
        cfg_init = config if config is not None else kwargs.get("config", {})
        def _resolve_arg(name, default_val=None, is_enum=False, enum_cls=None):
            val_kwarg = kwargs.get(name); val_cfg = cfg_init.get(name)
            val = val_kwarg if val_kwarg is not None else val_cfg
            if val is None and default_val is not None: val = default_val
            if is_enum and isinstance(val, str) and enum_cls is not None: return enum_cls.from_string(val)
            return val
        init_args = {
            "activation_dim": _resolve_arg("activation_dim"), "dict_size": _resolve_arg("dict_size"),
            "num_encoder_layers": _resolve_arg("num_encoder_layers"),
            "decoder_type": _resolve_arg("decoder_type", "linear"),
            "same_init_for_all_layers": _resolve_arg("same_init_for_all_layers", False),
            "norm_init_scale": _resolve_arg("norm_init_scale"),
            "init_with_transpose": _resolve_arg("init_with_transpose", True),
            "encoder_layers_indices": _resolve_arg("encoder_layers_indices"),
            "num_decoder_output_layers": _resolve_arg("num_decoder_output_layers"),
            "code_normalization": _resolve_arg("code_normalization", CodeNormalization.CROSSCODER, True, CodeNormalization),
            "code_normalization_alpha_sae": _resolve_arg("code_normalization_alpha_sae", 1.0),
            "code_normalization_alpha_cc": _resolve_arg("code_normalization_alpha_cc", 0.1),}
        if 'latent_processor' in kwargs: init_args['latent_processor'] = kwargs['latent_processor']
        if not all([init_args["activation_dim"], init_args["dict_size"], init_args["num_encoder_layers"]]):
            if not from_hub and path_or_model_id:
                try:
                    sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                    if "_hub_mixin_config" in sd : cfg_init.update(sd["_hub_mixin_config"])
                    for name in ["activation_dim", "dict_size", "num_encoder_layers", "decoder_type",
                                 "encoder_layers_indices", "num_decoder_output_layers", "code_normalization"]:
                        if init_args.get(name) is None:
                            val = cfg_init.get(name)
                            if name == "code_normalization" and isinstance(val, str): val = CodeNormalization.from_string(val)
                            elif name == "code_normalization" and val is None: val = CodeNormalization.CROSSCODER
                            init_args[name] = val
                    if not all([init_args["activation_dim"], init_args["dict_size"], init_args["num_encoder_layers"]]):
                        num_enc_mats_sd, act_dim_sd, dict_size_sd = sd["encoder.weight"].shape
                        if init_args["activation_dim"] is None: init_args["activation_dim"] = act_dim_sd
                        if init_args["dict_size"] is None: init_args["dict_size"] = dict_size_sd
                        if init_args["num_encoder_layers"] is None:
                            eli_sd = init_args.get("encoder_layers_indices")
                            init_args["num_encoder_layers"] = len(eli_sd) if eli_sd else num_enc_mats_sd
                except Exception as e:
                    raise ValueError(f"Could not load/infer params from local {path_or_model_id} for CrossCoder: {e}")
            if not all([init_args["activation_dim"], init_args["dict_size"], init_args["num_encoder_layers"]]):
                raise ValueError("activation_dim, dict_size, num_encoder_layers must be provided or inferable.")
        final_cfg_for_hub = cfg_init.copy(); final_cfg_for_hub.update(init_args)
        if isinstance(final_cfg_for_hub.get("code_normalization"), CodeNormalization):
             final_cfg_for_hub["code_normalization"] = str(final_cfg_for_hub["code_normalization"])
        if 'latent_processor' in final_cfg_for_hub: del final_cfg_for_hub['latent_processor']
        if from_hub:
            model = super(CrossCoder, cls).from_pretrained(path_or_model_id, config=final_cfg_for_hub, **kwargs) # type: ignore
        else:
            instance = cls(**init_args) # type: ignore
            state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
            if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
            instance.load_state_dict(state_dict)
            model = instance
        model.to(dtype=dtype); # type: ignore
        if device: model.to(device); # type: ignore
        return model # type: ignore

class BatchTopKCrossCoder(CrossCoder):
    def __init__(
        self, activation_dim: int, dict_size: int, num_encoder_layers: int,
        k: Union[int, th.Tensor] = 100,
        norm_init_scale: float = 1.0,
        decoder_type: str = "linear",
        **kwargs, ):
        code_norm_kwarg = kwargs.get('code_normalization', CodeNormalization.CROSSCODER)
        resolved_code_norm = CodeNormalization.from_string(code_norm_kwarg) if isinstance(code_norm_kwarg, str) else code_norm_kwarg
        kwargs['code_normalization'] = resolved_code_norm
        super().__init__(
            activation_dim, dict_size, num_encoder_layers,
            decoder_type=decoder_type, norm_init_scale=norm_init_scale, **kwargs)
        k_val = k if isinstance(k,int) else k.item()
        self._hub_mixin_config.update({"k": k_val})
        self.k_tensor = th.tensor(k_val, dtype=th.int) if not isinstance(k, th.Tensor) else k.to(th.int)
        self.register_buffer("k", self.k_tensor)
        thresh_init_val = [-1.0]*self.num_encoder_layers_processed if self.decoupled_code else -1.0
        self.register_buffer("threshold", th.tensor(thresh_init_val, dtype=th.float32))

    def encode(
        self, x: th.Tensor, return_active: bool = False, use_threshold: bool = True,
        select_features: Optional[list[int]] = None,):
        batch_size = x.size(0); k_act = self.k.item()
        f_summed_post_relu = super().encode(x, select_features=select_features)
        current_dict_size = f_summed_post_relu.shape[1]
        code_norm = self.get_code_normalization(select_features=select_features)
        f_actual_for_decode: th.Tensor; f_scaled_for_aux_log: th.Tensor; raw_f_scaled_for_aux_log: th.Tensor
        if self.decoupled_code:
            f_scaled_per_dec_layer = f_summed_post_relu.unsqueeze(1) * code_norm.unsqueeze(0)
            f_for_decode_per_layer = th.zeros_like(f_scaled_per_dec_layer)
            active_threshold = (self.threshold >= 0).any().item() if self.threshold.numel() > 1 else (self.threshold.item() >=0)
            if use_threshold and active_threshold:
                mask = f_scaled_per_dec_layer > self.threshold.view(1,-1,1)
                f_for_decode_per_layer = f_summed_post_relu.unsqueeze(1) * mask
            else:
                for l_idx in range(self.decoder.num_output_layers):
                    l_scaled_feats = f_scaled_per_dec_layer[:,l_idx,:]
                    num_sel = min(k_act * batch_size, l_scaled_feats.numel())
                    if num_sel <= 0: continue
                    topk_idx_flat = l_scaled_feats.flatten().topk(num_sel, sorted=False).indices
                    vals_orig_flat = f_summed_post_relu.flatten()
                    scatter_vals = vals_orig_flat[topk_idx_flat]
                    temp_flat = th.zeros_like(l_scaled_feats.flatten()).scatter_(-1, topk_idx_flat, scatter_vals)
                    f_for_decode_per_layer[:,l_idx,:] = temp_flat.reshape(batch_size, current_dict_size)
            f_actual_for_decode = f_for_decode_per_layer
            f_scaled_for_aux_log = f_scaled_per_dec_layer.sum(dim=1)
            raw_f_scaled_for_aux_log = f_scaled_per_dec_layer
        else:
            f_scaled = f_summed_post_relu * code_norm
            active_threshold = self.threshold.item() >= 0
            if use_threshold and active_threshold:
                mask = f_scaled > self.threshold
                f_actual_for_decode = f_summed_post_relu * mask
            else:
                num_sel = min(k_act * batch_size, f_scaled.numel())
                if num_sel <= 0: f_actual_for_decode = th.zeros_like(f_scaled)
                else:
                    topk_idx_flat = f_scaled.flatten().topk(num_sel, sorted=False).indices
                    vals_orig_flat = f_summed_post_relu.flatten()
                    scatter_vals = vals_orig_flat[topk_idx_flat]
                    temp_flat = th.zeros_like(f_scaled.flatten()).scatter_(-1, topk_idx_flat, scatter_vals)
                    f_actual_for_decode = temp_flat.reshape(batch_size, current_dict_size)
            f_scaled_for_aux_log = f_scaled; raw_f_scaled_for_aux_log = f_scaled
        if return_active:
            active_sum_dims = (0,1) if f_actual_for_decode.dim() == 3 else 0
            active_F = (f_actual_for_decode.sum(dim=active_sum_dims) > 0)
            active_indices_F = active_F.nonzero(as_tuple=True)[0]
            return (f_actual_for_decode, f_scaled_for_aux_log, active_indices_F, f_summed_post_relu, raw_f_scaled_for_aux_log)
        else: return f_actual_for_decode

    @classmethod
    def from_pretrained(cls, path_or_model_id, dtype=th.float32, device=None, from_hub=False, config=None, **kwargs):
        cfg_init = config if config is not None else kwargs.get("config", {})
        def _resolve_arg_btkcc(name, default_val=None, is_enum=False, enum_cls=None):
            val_kwarg = kwargs.get(name); val_cfg = cfg_init.get(name)
            val = val_kwarg if val_kwarg is not None else val_cfg
            if val is None and default_val is not None: val = default_val
            if is_enum and isinstance(val, str) and enum_cls is not None: return enum_cls.from_string(val)
            return val
        init_args_btkcc = {
            "activation_dim": _resolve_arg_btkcc("activation_dim"), "dict_size": _resolve_arg_btkcc("dict_size"),
            "num_encoder_layers": _resolve_arg_btkcc("num_encoder_layers"), "k": _resolve_arg_btkcc("k", 100),
            "norm_init_scale": _resolve_arg_btkcc("norm_init_scale", 1.0),
            "decoder_type": _resolve_arg_btkcc("decoder_type", "linear"),}
        cc_arg_names = ["same_init_for_all_layers", "init_with_transpose", "encoder_layers_indices",
                        "num_decoder_output_layers", "code_normalization",
                        "code_normalization_alpha_sae", "code_normalization_alpha_cc", "latent_processor"]
        for name in cc_arg_names: init_args_btkcc[name] = _resolve_arg_btkcc(name)
        if not all([init_args_btkcc["activation_dim"], init_args_btkcc["dict_size"], init_args_btkcc["num_encoder_layers"]]):
            if not from_hub and path_or_model_id:
                try:
                    sd = th.load(path_or_model_id, map_location='cpu', weights_only=True)
                    if "_hub_mixin_config" in sd : cfg_init.update(sd["_hub_mixin_config"])
                    for name_key in init_args_btkcc:
                        is_enum = name_key == "code_normalization"; enum_c = CodeNormalization if is_enum else None
                        default = 100 if name_key == "k" else (1.0 if name_key == "norm_init_scale" else ("linear" if name_key == "decoder_type" else None))
                        init_args_btkcc[name_key] = _resolve_arg_btkcc(name_key, default, is_enum, enum_c)
                    if not all([init_args_btkcc["activation_dim"], init_args_btkcc["dict_size"], init_args_btkcc["num_encoder_layers"]]):
                        num_enc_mats_sd, act_dim_sd, dict_size_sd = sd["encoder.weight"].shape
                        if init_args_btkcc["activation_dim"] is None: init_args_btkcc["activation_dim"] = act_dim_sd
                        if init_args_btkcc["dict_size"] is None: init_args_btkcc["dict_size"] = dict_size_sd
                        if init_args_btkcc["num_encoder_layers"] is None:
                            eli_sd = init_args_btkcc.get("encoder_layers_indices")
                            init_args_btkcc["num_encoder_layers"] = len(eli_sd) if eli_sd else num_enc_mats_sd
                except Exception as e:
                     raise ValueError(f"Could not load/infer params from local {path_or_model_id} for BTKCC: {e}")
            if not all([init_args_btkcc["activation_dim"], init_args_btkcc["dict_size"], init_args_btkcc["num_encoder_layers"], init_args_btkcc.get("k") is not None]):
                raise ValueError("activation_dim, dict_size, num_encoder_layers, k must be provided or inferable.")
        final_cfg_for_hub = cfg_init.copy(); final_cfg_for_hub.update(init_args_btkcc)
        if isinstance(final_cfg_for_hub.get("code_normalization"), CodeNormalization):
            final_cfg_for_hub["code_normalization"] = str(final_cfg_for_hub["code_normalization"])
        if 'latent_processor' in final_cfg_for_hub: del final_cfg_for_hub['latent_processor']
        super_kwargs_init = {k:v for k,v in kwargs.items() if k not in init_args_btkcc}
        if from_hub:
            model = super(BatchTopKCrossCoder, cls).from_pretrained(path_or_model_id, config=final_cfg_for_hub, **super_kwargs_init)
        else:
            final_init_args_local = init_args_btkcc.copy()
            if isinstance(final_init_args_local.get("code_normalization"), str):
                 final_init_args_local["code_normalization"] = CodeNormalization.from_string(final_init_args_local["code_normalization"])
            instance = cls(**final_init_args_local) # type: ignore
            state_dict = th.load(path_or_model_id, map_location=device if device else 'cpu', weights_only=True)
            if "_hub_mixin_config" in state_dict: del state_dict["_hub_mixin_config"]
            instance.load_state_dict(state_dict)
            model = instance
        model.to(dtype=dtype); # type: ignore
        if device: model.to(device); # type: ignore
        return model # type: ignore
