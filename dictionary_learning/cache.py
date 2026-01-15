import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from nnsight import LanguageModel
from typing import Tuple, List
import numpy as np
import os
from tqdm.auto import tqdm
from multiprocessing import Pool, Manager
import time
import json
from .config import DEBUG
from .utils import (
    dtype_to_str,
    str_to_dtype,
    torch_to_numpy_dtype,
)

if DEBUG:
    tracer_kwargs = {"scan": True, "validate": True}
else:
    tracer_kwargs = {"scan": False, "validate": False}

import torch
from typing import Tuple


class RunningStatWelford:
    """
    Streaming (online) mean / variance with Welford's algorithm.

    Works for arbitrary feature shapes – e.g. a vector of size D, a 2-D image
    channel grid, … anything except that the first axis of the update batch
    is interpreted as the *sample* axis.

    Args:
        shape: Feature shape tuple (e.g., (512,) for vector features)
        dtype: Data type for internal computations
        device: Device to store tensors on

    Example:
        stats = RunningStatWelford(shape=(512,), device="cuda")
        for batch in dataloader:
            stats.update(batch)          # batch.shape == (B, 512)

        print(stats.mean)                # current running mean
        print(stats.std(unbiased=True))  # sample std-dev (Bessel-corrected)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype=torch.float64,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        self.count = torch.tensor(0, dtype=torch.long, device=self.device)
        self.mean = torch.zeros(shape, dtype=dtype, device=self.device)
        self.M2 = torch.zeros(shape, dtype=dtype, device=self.device)

    def save_state(self, store_dir: str):
        """
        Save the current state of the running statistics to a file.
        """
        torch.save(self.count.cpu(), os.path.join(store_dir, "count.pt"))
        torch.save(self.mean.cpu(), os.path.join(store_dir, "mean.pt"))
        torch.save(self.M2.cpu(), os.path.join(store_dir, "M2.pt"))

    @staticmethod
    def load_or_create_state(
        store_dir: str,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        shape: Tuple[int, ...] = None,
    ):
        """
        Load the current state of the running statistics from a file.
        """
        if os.path.exists(os.path.join(store_dir, "count.pt")):
            count = torch.load(
                os.path.join(store_dir, "count.pt"),
                weights_only=True,
                map_location=device,
            )
            mean = torch.load(
                os.path.join(store_dir, "mean.pt"),
                weights_only=True,
                map_location=device,
            )
            M2 = torch.load(
                os.path.join(store_dir, "M2.pt"), weights_only=True, map_location=device
            )
            stat = RunningStatWelford(
                shape=mean.shape,
                dtype=mean.dtype,
                device=device,
            )
            stat.count = count
            stat.mean = mean
            stat.M2 = M2
            return stat
        else:
            return RunningStatWelford(shape=shape, dtype=dtype, device=device)

    def update(self, x: torch.Tensor) -> None:
        """
        Incorporate a new mini-batch `x` whose *first* dimension is batch-size.

        Args:
            x: Input tensor with batch dimension first
        """
        if x.numel() == 0:
            return  # nothing to do

        # ensure dtype/device match internal buffers
        x = x.clone().to(device=self.device, dtype=self.dtype)

        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)

        delta = batch_mean - self.mean
        total_n = self.count + batch_n

        # merge step (Chan-Golub-LeVeque)
        self.mean += delta * batch_n / total_n
        self.M2 += batch_M2 + (delta**2) * self.count * batch_n / total_n
        self.count = total_n

    def merge(self, other: "RunningStatWelford") -> None:
        """
        Merge another (independent) accumulator into this one in O(1).
        Useful for distributed training / multi-loader aggregation.

        Args:
            other: Another RunningStatWelford instance to merge
        """
        if other.count == 0:
            return
        if self.count == 0:
            # shallow copy of buffers
            self.count = other.count.clone()
            self.mean = other.mean.clone()
            self.M2 = other.M2.clone()
            return

        delta = other.mean - self.mean
        total_n = self.count + other.count

        self.mean += delta * other.count / total_n
        self.M2 += other.M2 + (delta**2) * self.count * other.count / total_n
        self.count = total_n

    def var(self, unbiased: bool = True) -> torch.Tensor:
        """
        Return per-feature variance.

        Args:
            unbiased: If True, divide by (n-1) for sample variance (Bessel-corrected).
                     If False, divide by n for population variance.

        Returns:
            Per-feature variance tensor
        """
        if self.count < (2 if unbiased else 1):
            return torch.full_like(self.mean, float("nan"))
        denom = self.count - 1 if unbiased else self.count
        return self.M2 / denom

    def std(self, unbiased: bool = True) -> torch.Tensor:
        """
        Standard deviation (sqrt of `var`).

        Args:
            unbiased: If True, use sample std-dev. If False, use population std-dev.

        Returns:
            Per-feature standard deviation tensor
        """
        return torch.sqrt(self.var(unbiased=unbiased))

    @property
    def n(self) -> int:
        """Number of samples processed."""
        return int(self.count.item())


class ActivationShard:
    def __init__(
        self,
        store_dir: str,
        shard_idx: int,
    ):
        self.shard_file = os.path.join(store_dir, f"shard_{shard_idx}.memmap")
        with open(self.shard_file.replace(".memmap", ".meta"), "r") as f:
            meta = json.load(f)
            self.shape = tuple(meta["shape"])
            self.dtype = str_to_dtype(meta["dtype"]) if "dtype" in meta else th.float32
            if self.dtype == th.bfloat16:
                np_dtype = np.int16
            else:
                np_dtype = torch_to_numpy_dtype(self.dtype)
        self.activations = np.memmap(
            self.shard_file, dtype=np_dtype, mode="r", shape=self.shape
        )

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            data = self.activations[index]
        elif isinstance(index, tuple):
            data = self.activations[index]
        else:
            data = self.activations[index]
        return th.tensor(data).view(self.dtype)


def save_shard(activations, store_dir, shard_count, name, io):
    print(f"Storing activation shard ({activations.shape})")
    memmap_file = os.path.join(store_dir, f"shard_{shard_count}.memmap")
    memmap_file_meta = memmap_file.replace(".memmap", ".meta")
    dtype = activations.dtype
    if dtype == th.bfloat16:
        activations = activations.view(th.int16)
    activations = activations.numpy()
    memmap = np.memmap(
        memmap_file,
        dtype=activations.dtype,
        mode="w+",
        shape=(activations.shape[0], activations.shape[1]),
    )
    memmap[:] = activations
    memmap.flush()
    with open(memmap_file_meta, "w") as f:
        json.dump({"shape": list(activations.shape), "dtype": dtype_to_str(dtype)}, f)
    del memmap
    print(f"Finished storing activations for shard {shard_count}")


class ActivationCache:
    __pool = None
    __active_processes = None
    __process_lock = None
    __manager = None

    def __init__(self, store_dir: str, submodule_name: str = None):
        if submodule_name is None:
            import warnings

            warnings.warn(
                "submodule_name parameter will be required in future versions. "
                "Please specify the submodule name when creating ActivationCache instances and specify the store_dir without the submodule folder.",
                FutureWarning,
                stacklevel=2,
            )
            self._cache_store_dir = store_dir
        else:
            self._cache_store_dir = os.path.join(store_dir, submodule_name)

        self.config = json.load(
            open(os.path.join(self._cache_store_dir, "config.json"), "r")
        )
        self.shards = [
            ActivationShard(self._cache_store_dir, i)
            for i in range(self.config["shard_count"])
        ]
        self._range_to_shard_idx = np.cumsum([0] + [s.shape[0] for s in self.shards])
        if "store_tokens" in self.config and self.config["store_tokens"]:
            self._tokens = th.load(
                os.path.join(store_dir, "tokens.pt"),
                weights_only=True,
                map_location=th.device("cpu"),
            )

        self._sequence_ranges = None
        self._mean = None
        self._std = None

    @property
    def mean(self):
        if self._mean is None:
            if os.path.exists(os.path.join(self._cache_store_dir, "mean.pt")):
                self._mean = th.load(
                    os.path.join(self._cache_store_dir, "mean.pt"),
                    weights_only=True,
                    map_location=th.device("cpu"),
                )
            else:
                raise ValueError(
                    f"Mean not found for {self._cache_store_dir}. Re-run the collection script."
                )
        return self._mean

    @property
    def std(self):
        if self._std is None:
            if os.path.exists(os.path.join(self._cache_store_dir, "std.pt")):
                self._std = th.load(
                    os.path.join(self._cache_store_dir, "std.pt"),
                    weights_only=True,
                    map_location=th.device("cpu"),
                )
            else:
                raise ValueError(
                    f"Std not found for {self._cache_store_dir}. Re-run the collection script."
                )
        return self._std

    @property
    def running_stats(self):
        return RunningStatWelford.load_or_create_state(
            self._cache_store_dir, shape=(self.config["d_model"],)
        )

    def __len__(self):
        return self.config["total_size"]

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Handle slice objects
            start, stop, step = index.indices(len(self))
            start_shard_idx = (
                np.searchsorted(self._range_to_shard_idx, start, side="right") - 1
            )
            stop_shard_idx = (
                np.searchsorted(self._range_to_shard_idx, stop, side="right") - 1
            )
            if start_shard_idx == stop_shard_idx:
                offset = start - self._range_to_shard_idx[start_shard_idx]
                end_offset = stop - self._range_to_shard_idx[stop_shard_idx]
                shard = self.shards[start_shard_idx]
                return shard[offset:end_offset:step]
            else:
                # Lazily load if we are not in the same shard
                # TODO: make this more efficient
                return th.stack([self[i] for i in range(start, stop, step)], dim=0)
        elif isinstance(index, int):
            # Handle single integer index
            shard_idx = (
                np.searchsorted(self._range_to_shard_idx, index, side="right") - 1
            )
            offset = index - self._range_to_shard_idx[shard_idx]
            shard = self.shards[shard_idx]
            return shard[offset]
        elif isinstance(index, th.Tensor):
            # Handle torch.Tensor index (convert to int if scalar)
            if index.numel() == 1:
                return self[index.item()]
            else:
                raise TypeError(f"Tensor index must be scalar, got shape {index.shape}")
        else:
            raise TypeError(
                f"Index must be int, slice, or scalar tensor, got {type(index)}"
            )

    @property
    def tokens(self):
        return self._tokens

    @property
    def sequence_ranges(self):
        if hasattr(self, "_sequence_ranges") and self._sequence_ranges is not None:
            return self._sequence_ranges

        if (
            "store_sequence_ranges" in self.config
            and self.config["store_sequence_ranges"]
            and os.path.exists(
                os.path.join(self._cache_store_dir, "..", "sequence_ranges.pt")
            )
        ):
            self._sequence_ranges = th.load(
                os.path.join(self._cache_store_dir, "..", "sequence_ranges.pt"),
                weights_only=True,
            ).cpu()
            return self._sequence_ranges
        else:
            # Return None if sequence ranges not available
            return None

    @staticmethod
    def get_activations(submodule: nn.Module, io: str):
        if io == "in":
            x = submodule.input
        else:
            x = submodule.output
        if isinstance(x, tuple):
            return x[0]
        return x

    @staticmethod
    def __init_multiprocessing(max_concurrent_saves: int = 3):
        if ActivationCache.__pool is None:
            ActivationCache.__manager = Manager()
            ActivationCache.__active_processes = ActivationCache.__manager.Value("i", 0)
            ActivationCache.__process_lock = ActivationCache.__manager.Lock()
            ActivationCache.__pool = Pool(processes=max_concurrent_saves)

    @staticmethod
    def cleanup_multiprocessing():
        if ActivationCache.__pool is not None:
            # wait for all processes to finish
            while ActivationCache.__active_processes.value > 0:
                print(
                    f"Waiting for {ActivationCache.__active_processes.value} save processes to finish"
                )
                time.sleep(10)
            ActivationCache.__pool.close()
            ActivationCache.__pool = None
            ActivationCache.__manager.shutdown()
            ActivationCache.__manager = None
            ActivationCache.__active_processes = None
            ActivationCache.__process_lock = None

    @staticmethod
    def collate_store_shards(
        store_dirs: Tuple[str],
        shard_count: int,
        activation_cache: List[th.Tensor],
        submodule_names: Tuple[str],
        shuffle_shards: bool = True,
        io: str = "out",
        multiprocessing: bool = True,
        max_concurrent_saves: int = 3,
    ):
        # Create a process pool if multiprocessing is enabled
        if multiprocessing and ActivationCache.__pool is None:
            ActivationCache.__init_multiprocessing(max_concurrent_saves)

        if multiprocessing:
            pool = ActivationCache.__pool
            active_processes = ActivationCache.__active_processes
            process_lock = ActivationCache.__process_lock

        for i, name in enumerate(submodule_names):
            activations = th.cat(
                activation_cache[i], dim=0
            )  # (N x B x T) x D (N = number of batches per shard)

            if shuffle_shards:
                idx = np.random.permutation(activations.shape[0])
                activations = activations[idx]

            if multiprocessing:
                # Wait if we've reached max concurrent processes
                while active_processes.value >= max_concurrent_saves:
                    time.sleep(0.1)

                # Increment active process count
                with process_lock:
                    active_processes.value += 1

                def callback(result):
                    with process_lock:
                        active_processes.value -= 1

                print(
                    f"Applying async save for shard {shard_count} (current num of workers: {active_processes.value})"
                )
                pool.apply_async(
                    save_shard,
                    args=(activations, store_dirs[i], shard_count, name, io),
                    callback=callback,
                )
            else:
                save_shard(activations, store_dirs[i], shard_count, name, io)

    @staticmethod
    def shard_exists(store_dir: str, shard_count: int):
        if os.path.exists(os.path.join(store_dir, f"shard_{shard_count}.memmap")):
            # load the meta file
            with open(os.path.join(store_dir, f"shard_{shard_count}.meta"), "r") as f:
                shape = json.load(f)["shape"]
            return shape
        else:
            return None

    @staticmethod
    def exists(
        store_dir: str, submodule_names: Tuple[str], io: str, store_tokens: bool
    ):
        """
        Check if cached activations exist for the given configuration.

        Args:
            store_dir: Base directory where cached activations are stored
            submodule_names: Names of the submodules to check for cached activations
            io: Input/output type ("in" or "out") specifying which activations to check
            store_tokens: Whether tokens should also be stored and checked for existence

        Returns:
            Tuple[bool, int]: (exists, num_tokens) where exists indicates if all required
            cached data is present and num_tokens is the total number of tokens in the cache
        """
        num_tokens = 0
        config = None
        for submodule_name in submodule_names:
            config_path = os.path.join(
                store_dir, f"{submodule_name}_{io}", "config.json"
            )
            if not os.path.exists(config_path):
                return False, 0
            with open(config_path, "r") as f:
                config = json.load(f)
                num_tokens = config["total_size"]

        if store_tokens and not os.path.exists(os.path.join(store_dir, "tokens.pt")):
            return False, 0

        # Check for sequence ranges if they should exist
        if (
            config
            and "store_sequence_ranges" in config
            and config["store_sequence_ranges"]
            and not os.path.exists(os.path.join(store_dir, "sequence_ranges.pt"))
        ):
            return False, 0

        return True, num_tokens

    @th.no_grad()
    @staticmethod
    def collect(
        data: Dataset,
        submodules: Tuple[nn.Module],
        submodule_names: Tuple[str],
        model: LanguageModel,
        store_dir: str,
        batch_size: int = 64,
        context_len: int = 128,
        shard_size: int = 10**6,
        d_model: int = 1024,
        shuffle_shards: bool = False,
        io: str = "out",
        num_workers: int = 8,
        max_total_tokens: int = 10**8,
        last_submodule: nn.Module = None,
        overwrite: bool = False,
        store_tokens: bool = False,
        multiprocessing: bool = True,
        ignore_first_n_tokens_per_sample: int = 0,
        token_level_replacement: dict = None,
        add_special_tokens: bool = True,
        dtype: th.dtype = None,
    ):
        assert (
            not shuffle_shards or not store_tokens
        ), "Shuffling shards and storing tokens is not supported yet"

        store_sequence_ranges = store_tokens and not shuffle_shards

        dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers)

        activation_cache = [[] for _ in submodules]
        tokens_cache = []
        sequence_ranges_cache = []
        current_token_position = 0  # Track position in flattened token stream

        store_sub_dirs = [
            os.path.join(store_dir, f"{submodule_names[i]}_{io}")
            for i in range(len(submodules))
        ]
        for store_sub_dir in store_sub_dirs:
            os.makedirs(store_sub_dir, exist_ok=True)

        # load running stats
        running_stats = [
            RunningStatWelford.load_or_create_state(
                store_sub_dir, dtype, shape=(d_model,)
            )
            for store_sub_dir in store_sub_dirs
        ]

        total_size = 0
        current_size = 0
        shard_count = 0
        if ignore_first_n_tokens_per_sample > 0:
            model.tokenizer.padding_side = "right"

        print("Collecting activations...")
        for batch in tqdm(dataloader, desc="Collecting activations"):
            tokens = model.tokenizer(
                batch,
                max_length=context_len,
                truncation=True,
                return_tensors="pt",
                padding=True,
                add_special_tokens=add_special_tokens,
            )  # (B, T)

            if token_level_replacement is not None:
                # Iterate through the replacement dictionary and apply replacements efficiently
                new_ids = tokens[
                    "input_ids"
                ].clone()  # Clone to avoid modifying the original tensor if needed elsewhere
                for old_token_id, new_token_id in token_level_replacement.items():
                    # Create a mask for elements equal to the old_token_id
                    mask = new_ids == old_token_id
                    # Use the mask to update elements with the new_token_id
                    new_ids[mask] = new_token_id
                tokens["input_ids"] = new_ids

            attention_mask = tokens["attention_mask"]

            store_mask = attention_mask.clone()
            if ignore_first_n_tokens_per_sample > 0:
                store_mask[:, :ignore_first_n_tokens_per_sample] = 0

            # Track sequence ranges if needed
            if store_sequence_ranges:
                batch_lengths = store_mask.sum(dim=1).tolist()
                batch_sequence_ranges = (
                    np.cumsum([0] + batch_lengths[:-1]) + current_token_position
                )
                sequence_ranges_cache.extend(batch_sequence_ranges.tolist())
                current_token_position += sum(batch_lengths)

            if store_tokens:
                tokens_cache.append(
                    tokens["input_ids"].reshape(-1)[store_mask.reshape(-1).bool()]
                )

            # Check all store_sub_dirs and ensure they have the same shape
            shapes = [
                ActivationCache.shard_exists(store_sub_dir, shard_count)
                for store_sub_dir in store_sub_dirs
            ]
            if all(s is not None for s in shapes) and all(
                s == shapes[0] for s in shapes
            ):
                shape = shapes[0]
            else:
                shape = None
            if overwrite or shape is None:
                with model.trace(
                    tokens,
                    **tracer_kwargs,
                ):
                    for i, submodule in enumerate(submodules):
                        local_activations = (
                            ActivationCache.get_activations(submodule, io)
                            .reshape(-1, d_model)
                            .save()
                        )  # (B x T) x D
                        activation_cache[i].append(local_activations)

                    if last_submodule is not None:
                        last_submodule.output.stop()

                for i in range(len(submodules)):
                    activation_cache[i][-1] = (
                        activation_cache[i][-1][store_mask.reshape(-1).bool()]
                        .cpu()
                    )  # remove padding tokens
                    running_stats[i].update(activation_cache[i][-1].view(-1, d_model))
                    if dtype is not None:
                        activation_cache[i][-1] = activation_cache[i][-1].to(dtype)

                if store_tokens:
                    assert len(tokens_cache[-1]) == activation_cache[0][-1].shape[0]
                assert activation_cache[0][-1].shape[0] == store_mask.sum().item()
                current_size += activation_cache[0][-1].shape[0]
            else:
                current_size += store_mask.sum().item()

            if current_size > shard_size:
                if shape is not None and not overwrite:
                    assert shape[0] == current_size
                    print(f"Shard {shard_count} already exists. Skipping.")
                else:
                    print(f"Storing shard {shard_count}...", flush=True)
                    ActivationCache.collate_store_shards(
                        store_sub_dirs,
                        shard_count,
                        activation_cache,
                        submodule_names,
                        shuffle_shards,
                        io,
                        multiprocessing=multiprocessing,
                    )
                    for i in range(len(submodules)):
                        running_stats[i].save_state(store_sub_dirs[i])
                shard_count += 1

                total_size += current_size
                current_size = 0
                activation_cache = [[] for _ in submodules]

            if total_size > max_total_tokens:
                print("Max total tokens reached. Stopping collection.")
                break

        if current_size > 0:
            if shape is not None and not overwrite:
                assert shape[0] == current_size
                print(f"Shard {shard_count} already exists. Skipping.")
            else:
                print(f"Storing shard {shard_count}...", flush=True)
                ActivationCache.collate_store_shards(
                    store_sub_dirs,
                    shard_count,
                    activation_cache,
                    submodule_names,
                    shuffle_shards,
                    io,
                    multiprocessing=multiprocessing,
                )
                for i in range(len(submodules)):
                    running_stats[i].save_state(store_sub_dirs[i])
            shard_count += 1
            total_size += current_size

        # store configs
        for i, store_sub_dir in enumerate(store_sub_dirs):
            with open(os.path.join(store_sub_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "batch_size": batch_size,
                        "context_len": context_len,
                        "shard_size": shard_size,
                        "d_model": d_model,
                        "shuffle_shards": shuffle_shards,
                        "io": io,
                        "total_size": total_size,
                        "shard_count": shard_count,
                        "store_tokens": store_tokens,
                        "store_sequence_ranges": store_sequence_ranges,
                    },
                    f,
                )

        # store tokens
        if store_tokens:
            print("Storing tokens...")
            tokens_cache = th.cat(tokens_cache, dim=0)
            assert (
                tokens_cache.shape[0] == total_size
            ), f"{tokens_cache.shape[0]} != {total_size}"
            th.save(tokens_cache, os.path.join(store_dir, "tokens.pt"))

        # store sequence ranges
        if store_sequence_ranges:
            print("Storing sequence ranges...")
            # add the last sequence range to the end of the cache
            sequence_ranges_cache.append(current_token_position)
            assert sequence_ranges_cache[-1] == total_size
            sequence_ranges_tensor = th.tensor(sequence_ranges_cache, dtype=th.long)
            th.save(
                sequence_ranges_tensor, os.path.join(store_dir, "sequence_ranges.pt")
            )
            print(f"Stored {len(sequence_ranges_cache)} sequence ranges")

        # store running stats
        for i in range(len(submodules)):
            th.save(
                running_stats[i].mean.cpu(), os.path.join(store_sub_dirs[i], "mean.pt")
            )
            th.save(
                running_stats[i].std().cpu(), os.path.join(store_sub_dirs[i], "std.pt")
            )

        ActivationCache.cleanup_multiprocessing()
        print(f"Finished collecting activations. Total size: {total_size}")


class PairedActivationCache:
    def __init__(self, store_dir_1: str, store_dir_2: str, submodule_name: str = None):
        self.activation_cache_1 = ActivationCache(store_dir_1, submodule_name)
        self.activation_cache_2 = ActivationCache(store_dir_2, submodule_name)
        if len(self.activation_cache_1) != len(self.activation_cache_2):
            min_len = min(len(self.activation_cache_1), len(self.activation_cache_2))
            assert (
                self.activation_cache_1.tokens is not None
                and self.activation_cache_2.tokens is not None
            ), "Caches have not the same length and tokens are not stored"
            assert torch.all(
                self.activation_cache_1.tokens[:min_len]
                == self.activation_cache_2.tokens[:min_len]
            ), "Tokens do not match"
            self._len = min_len
            print(
                f"Warning: Caches have not the same length. Using the first {min_len} tokens."
            )
            if len(self.activation_cache_1) > self._len:
                self._sequence_ranges = self.activation_cache_2.sequence_ranges
            else:
                self._sequence_ranges = self.activation_cache_1.sequence_ranges
        else:
            assert len(self.activation_cache_1) == len(
                self.activation_cache_2
            ), f"Lengths do not match: {len(self.activation_cache_1)} != {len(self.activation_cache_2)}"
            self._len = len(self.activation_cache_1)
            self._sequence_ranges = self.activation_cache_1.sequence_ranges

        if (
            self.activation_cache_1.tokens is not None
            and self.activation_cache_2.tokens is not None
        ):
            assert torch.all(
                self.activation_cache_1.tokens[: self._len]
                == self.activation_cache_2.tokens[: self._len]
            ), "Tokens do not match"

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if isinstance(index, slice):
            return th.stack(
                (self.activation_cache_1[index], self.activation_cache_2[index]), dim=1
            )
        elif isinstance(index, int):
            return th.stack(
                (self.activation_cache_1[index], self.activation_cache_2[index]), dim=0
            )
        elif isinstance(index, th.Tensor):
            return th.stack(
                (self.activation_cache_1[index], self.activation_cache_2[index]), dim=0
            )
        else:
            raise TypeError(f"Index must be int or slice, got {type(index)}")

    @property
    def tokens(self):
        return th.stack(
            (
                self.activation_cache_1.tokens[: self._len],
                self.activation_cache_2.tokens[: self._len],
            ),
            dim=0,
        )

    @property
    def sequence_ranges(self):
        return self._sequence_ranges

    @property
    def mean(self):
        return th.stack(
            (self.activation_cache_1.mean, self.activation_cache_2.mean), dim=0
        )

    @property
    def std(self):
        return th.stack(
            (self.activation_cache_1.std, self.activation_cache_2.std), dim=0
        )


class ActivationCacheTuple:
    def __init__(self, *store_dirs: str, submodule_name: str = None):
        self.activation_caches = [
            ActivationCache(store_dir, submodule_name) for store_dir in store_dirs
        ]
        assert len(self.activation_caches) > 0
        for i in range(1, len(self.activation_caches)):
            assert len(self.activation_caches[i]) == len(self.activation_caches[0])

    def __len__(self):
        return len(self.activation_caches[0])

    def __getitem__(self, index: int):
        return th.stack([cache[index] for cache in self.activation_caches], dim=0)

    @property
    def tokens(self):
        return th.stack([cache.tokens for cache in self.activation_caches], dim=0)

    @property
    def sequence_ranges(self):
        seq_starts_list = [cache.sequence_ranges for cache in self.activation_caches]
        if all(seq_starts is not None for seq_starts in seq_starts_list):
            return th.stack(seq_starts_list, dim=0)
        return None

    @property
    def mean(self):
        return th.stack([cache.mean for cache in self.activation_caches], dim=0)

    @property
    def std(self):
        return th.stack([cache.std for cache in self.activation_caches], dim=0)
