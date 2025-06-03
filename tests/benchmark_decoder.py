# tests/benchmark_decoder.py
import torch
import torch.nn as nn
import time
from typing import Type, Dict, Any, Optional, Generator

# Ensure dictionary_learning is accessible.
try:
    # These are the models we intend to benchmark with their default decoders
    from dictionary_learning.dictionary import BatchTopKSAE, BatchTopKCrossCoder
except ImportError:
    print("Error: Could not import from dictionary_learning. Ensure it's in PYTHONPATH.")
    raise

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ACTIVATION_DIM = 128
DICT_SIZE = 1024
K_SAE = 16
NUM_ENCODER_LAYERS_CC = 2 # For CrossCoder, this is the number of layers it's configured for
NUM_DECODER_LAYERS_CC = 2 # For CrossCoder, num of output layers for its decoder

BATCH_SIZE_TRAIN = 32
NUM_STEPS_TRAIN = 20
NUM_STEPS_WARMUP = 3
BATCH_SIZE_INFER = 32
NUM_STEPS_INFER = 30

# --- Helper Functions ---
def generate_synthetic_data(
    num_batches: int, batch_size: int, activation_dim: int, device: torch.device, num_layers: Optional[int] = None
) -> Generator[torch.Tensor, None, None]:
    for _ in range(num_batches):
        if num_layers is not None:
            yield torch.randn(batch_size, num_layers, activation_dim, device=device)
        else:
            yield torch.randn(batch_size, activation_dim, device=device)

# --- Simplified Model Instantiation Check ---
def check_model_instantiation(
    model_class: Type[nn.Module],
    model_name: str,
    model_args: Dict[str, Any], # Renamed from common_model_args for clarity
    device: torch.device
) -> bool:
    print(f"--- Checking Instantiation for {model_name} ---")
    try:
        model = model_class(**model_args).to(device)
        print(f"  Successfully instantiated {model_name}.")

        # Basic forward pass check
        is_cc = "CrossCoder" in model_name
        test_batch_size = 2
        activation_dim_check = model_args.get("activation_dim", ACTIVATION_DIM) # Use actual model arg if possible

        if is_cc:
            # Use num_encoder_layers (positional for BTKCC) or a default if not specified in args
            num_enc_l = model_args.get("num_encoder_layers", NUM_ENCODER_LAYERS_CC)
            test_input = next(generate_synthetic_data(1, test_batch_size, activation_dim_check, device, num_layers=num_enc_l))
        else:
            test_input = next(generate_synthetic_data(1, test_batch_size, activation_dim_check, device))

        model.eval()
        with torch.no_grad():
            _ = model(test_input)
        print(f"  Basic forward pass successful for {model_name}.")
        return True
    except Exception as e:
        print(f"  Error during instantiation or basic check for {model_name}: {e}")
        return False

# --- Main Benchmarking Function ---
def run_benchmarks():
    # Define base arguments for each model (without decoder_type)
    sae_args = {"activation_dim": ACTIVATION_DIM, "dict_size": DICT_SIZE, "k": K_SAE}

    # For BatchTopKCrossCoder, positional 'num_encoder_layers' and kwargs 'encoder_layers_indices', 'num_decoder_output_layers'
    # The dictionary.py now expects: __init__(self, activation_dim, dict_size, num_encoder_layers, k, ..., **kwargs)
    # where kwargs can include encoder_layers_indices, num_decoder_output_layers for the base CrossCoder
    btkcc_args = {
        "activation_dim": ACTIVATION_DIM,
        "dict_size": DICT_SIZE,
        "num_encoder_layers": NUM_ENCODER_LAYERS_CC, # This is the positional argument
        "k": K_SAE,
        # These are passed as **kwargs to BatchTopKCrossCoder -> CrossCoder
        "encoder_layers_indices": list(range(NUM_ENCODER_LAYERS_CC)),
        "num_decoder_output_layers": NUM_DECODER_LAYERS_CC,
    }

    # Simplified instantiation checks
    sae_instantiation_ok = check_model_instantiation(BatchTopKSAE, "BatchTopKSAE", sae_args, DEVICE)
    cc_instantiation_ok = check_model_instantiation(BatchTopKCrossCoder, "BatchTopKCrossCoder", btkcc_args, DEVICE)

    if not (sae_instantiation_ok and cc_instantiation_ok):
        print("!!! One or more models failed instantiation or basic check. Benchmark results might be incomplete or misleading. !!!")
        # Decide if to proceed or exit. For now, proceed.

    print("\n--- Starting Performance Benchmarks (Default Decoders) ---")
    model_configs_for_perf = {
        "BatchTopKSAE": (BatchTopKSAE, sae_args, False), # model_args, is_cross_coder_flag
        "BatchTopKCrossCoder": (BatchTopKCrossCoder, btkcc_args, True),
    }
    results: Dict[str, Dict[str, Any]] = {} # Type hint for clarity

    for model_name, (model_class, model_specific_args, is_cross_coder_flag) in model_configs_for_perf.items():
        print(f"\n-- Benchmarking {model_name} (Default Decoder) --")
        # Initialize results for this model configuration
        results[model_name] = {}

        try:
            # Instantiate model with its default decoder (no decoder_type passed)
            model = model_class(**model_specific_args).to(DEVICE)
        except Exception as e:
            print(f"Error instantiating {model_name}: {e}")
            results[model_name]["error_instantiation"] = str(e)
            continue # Skip to next model

        # Use AdamW, as it's generally robust.
        # The previous optimizer issues were related to EmbeddingBag(sparse=True).
        # Since we are now using default decoders, this should be fine.
        # (If default decoder IS EmbeddingBag(sparse=True), error will reappear here, but that's a model design issue then)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # --- Training Benchmark ---
        model.train()
        # Determine num_layers for synthetic data generation for CrossCoder
        num_train_data_layers = model_specific_args.get("num_encoder_layers") if is_cross_coder_flag else None

        train_data_loader = generate_synthetic_data(
            NUM_STEPS_TRAIN + NUM_STEPS_WARMUP, BATCH_SIZE_TRAIN,
            model_specific_args["activation_dim"], DEVICE,
            num_layers=num_train_data_layers
        )

        try:
            for _ in range(NUM_STEPS_WARMUP):
                x_batch = next(train_data_loader)
                optimizer.zero_grad()
                reconstruction = model(x_batch)
                loss = ((reconstruction - x_batch)**2).sum()
                loss.backward(); optimizer.step()

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start_time_train = time.perf_counter()
            for _ in range(NUM_STEPS_TRAIN):
                x_batch = next(train_data_loader)
                optimizer.zero_grad()
                reconstruction = model(x_batch)
                loss = ((reconstruction - x_batch)**2).sum()
                loss.backward(); optimizer.step()
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            end_time_train = time.perf_counter()
            avg_time_train = (end_time_train - start_time_train) / NUM_STEPS_TRAIN
            results[model_name]["train_step_time_ms"] = avg_time_train * 1000
            print(f"    Avg Training Step Time: {avg_time_train*1000:.3f} ms")
        except Exception as e:
            print(f"Error during training benchmark for {model_name}: {e}")
            results[model_name]["train_step_time_ms"] = "Error: " + str(e)

        # --- Inference Benchmarks ---
        model.eval()
        with torch.no_grad():
            # Determine num_layers for synthetic data for CrossCoder inference input
            num_infer_data_layers = model_specific_args.get("num_encoder_layers") if is_cross_coder_flag else None

            try: # Decode Only Benchmark
                sample_dense_input_for_encode = next(generate_synthetic_data(
                    1, BATCH_SIZE_INFER, model_specific_args["activation_dim"], DEVICE,
                    num_layers=num_infer_data_layers
                ))
                encode_output = model.encode(sample_dense_input_for_encode, return_active=True)
                features_for_decode_batch = (encode_output[0] if isinstance(encode_output, tuple) else encode_output).detach()

                for _ in range(NUM_STEPS_WARMUP): model.decode(features_for_decode_batch)
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                start_time_infer_decode = time.perf_counter()
                for _ in range(NUM_STEPS_INFER): model.decode(features_for_decode_batch)
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                end_time_infer_decode = time.perf_counter()
                avg_time_infer_decode = (end_time_infer_decode - start_time_infer_decode) / NUM_STEPS_INFER
                results[model_name]["infer_decode_time_ms"] = avg_time_infer_decode * 1000
                print(f"    Avg Inference (decode only) Time: {avg_time_infer_decode*1000:.3f} ms")
            except Exception as e:
                print(f"Error during decode-only inference for {model_name}: {e}")
                results[model_name]["infer_decode_time_ms"] = "Error: " + str(e)

            try: # Full Encode-Decode Benchmark
                infer_full_data_loader = generate_synthetic_data(
                    NUM_STEPS_INFER + NUM_STEPS_WARMUP, BATCH_SIZE_INFER,
                    model_specific_args["activation_dim"], DEVICE,
                    num_layers=num_infer_data_layers
                )
                for _ in range(NUM_STEPS_WARMUP): model(next(infer_full_data_loader))
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                start_time_infer_full = time.perf_counter()
                for _ in range(NUM_STEPS_INFER): model(next(infer_full_data_loader))
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                end_time_infer_full = time.perf_counter()
                avg_time_infer_full = (end_time_infer_full - start_time_infer_full) / NUM_STEPS_INFER
                results[model_name]["infer_full_model_time_ms"] = avg_time_infer_full * 1000
                print(f"    Avg Inference (encode-decode) Time: {avg_time_infer_full*1000:.3f} ms")
            except Exception as e:
                print(f"Error during full inference for {model_name}: {e}")
                results[model_name]["infer_full_model_time_ms"] = "Error: " + str(e)

    print("\n--- Benchmark Summary (Default Decoders) ---")
    for model_name_sum, model_results in results.items():
        print(f"  {model_name_sum}:")
        if isinstance(model_results, dict):
            for metric, t_val in model_results.items():
                if isinstance(t_val, float): print(f"      {metric}: {t_val:.3f} ms")
                else: print(f"      {metric}: {t_val}")
        else:
            print(f"    Error: {model_results}")

    print("--- Benchmarking Script Complete ---")

if __name__ == "__main__":
    run_benchmarks()
