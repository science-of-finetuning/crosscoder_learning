# tests/benchmark_decoder.py
# %% [markdown]
# # Decoder Benchmark Script
#
# This script benchmarks the performance of different decoder types (linear vs. EmbeddingBag)
# for `BatchTopKSAE` and `BatchTopKCrossCoder` models. It also includes a weight equivalence
# test to ensure numerical consistency between the decoder implementations.
# %%
# --- Imports and Setup ---
import torch
import torch.nn as nn
import time
from typing import Type, Dict, Any, Optional, Generator

try:
    from dictionary_learning.dictionary import BatchTopKSAE, BatchTopKCrossCoder
except ImportError:
    print("Error: Could not import from dictionary_learning. Ensure it's in PYTHONPATH.")
    raise

# %%
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ACTIVATION_DIM = 128
DICT_SIZE = 1024
K_SAE = 16
NUM_ENCODER_LAYERS_CC = 2
NUM_DECODER_LAYERS_CC = 2

BATCH_SIZE_TRAIN = 32
NUM_STEPS_TRAIN = 20
NUM_STEPS_WARMUP = 3
BATCH_SIZE_INFER = 32
NUM_STEPS_INFER = 30

# %%
# --- Helper Functions ---
def generate_synthetic_data(
    num_batches: int, batch_size: int, activation_dim: int, device: torch.device, num_layers: Optional[int] = None
) -> Generator[torch.Tensor, None, None]:
    for _ in range(num_batches):
        if num_layers is not None:
            yield torch.randn(batch_size, num_layers, activation_dim, device=device)
        else:
            yield torch.randn(batch_size, activation_dim, device=device)

# --- Weight Equivalence Test Function ---
@torch.no_grad()
def test_weight_equivalence(
    model_class: Type[nn.Module],
    model_name: str,
    common_model_args: Dict[str, Any],
    activation_dim: int,
    dict_size: int,
    device: torch.device
) -> bool:
    print(f"--- Testing Weight Equivalence for {model_name} ---")
    is_cc = "CrossCoder" in model_name

    args_linear = {**common_model_args, "decoder_type": "linear"}
    model_linear = model_class(**args_linear).to(device)

    args_eb = {**common_model_args, "decoder_type": "embedding_bag"}
    # Explicitly set use_sparse_decoder=True to match original intent, though it's the default
    args_eb_explicit = {**args_eb, "use_sparse_decoder": True}
    model_eb = model_class(**args_eb_explicit).to(device)

    model_eb.encoder.weight.data.copy_(model_linear.encoder.weight.data)
    if model_linear.encoder.bias is not None and model_eb.encoder.bias is not None:
        model_eb.encoder.bias.data.copy_(model_linear.encoder.bias.data)
    elif model_linear.encoder.bias is not None or model_eb.encoder.bias is not None:
        print("Warning: Encoder bias mismatch in existence between linear and EB models.")

    if hasattr(model_linear, 'b_dec') and hasattr(model_eb, 'b_dec'):
        model_eb.b_dec.data.copy_(model_linear.b_dec.data)
    elif hasattr(model_linear.decoder, 'bias') and hasattr(model_eb.decoder, 'bias'):
        model_eb.decoder.bias.data.copy_(model_linear.decoder.bias.data)
    else:
        print("Warning: Decoder bias (b_dec or decoder.bias) mismatch in existence.")

    if is_cc:
        if hasattr(model_linear.decoder, 'weight') and hasattr(model_eb.decoder, 'layers'):
            if not hasattr(model_linear.decoder, 'num_output_layers'): # num_output_layers is an attribute of CrossCoderDecoder
                print("Error: Linear CrossCoderDecoder missing 'num_output_layers' attribute for layer count.") # Should not happen if it's CrossCoderDecoder
                return False
            if len(model_eb.decoder.layers) != model_linear.decoder.num_output_layers:
                print(f"Error: EB CrossCoderDecoder layer count ({len(model_eb.decoder.layers)}) " +
                      f"mismatches Linear ({model_linear.decoder.num_output_layers}).")
                return False
            # Linear CrossCoderDecoder weight is (L,D,M), EB layers[i].weight is (D,M)
            # The new dictionary.py CrossCoderDecoder with EB stores weights directly in layers[i].weight as (D,M)
            # The linear CrossCoderDecoder has self.weight as (L,D,M)
            for i in range(model_linear.decoder.num_output_layers):
                 # model_linear.decoder.weight[i] is (D,M)
                 # model_eb.decoder.layers[i].weight is (D,M)
                model_eb.decoder.layers[i].weight.data.copy_(model_linear.decoder.weight.data[i].clone())
        else:
            print("Error: CrossCoder decoder structure mismatch for weight copy (linear.weight or eb.layers missing).")
            return False
    else: # BatchTopKSAE
        # model_linear.decoder (nn.Linear) weight is (act_dim, dict_size)
        # model_eb.decoder (nn.EmbeddingBag) weight is (dict_size, act_dim)
        if hasattr(model_linear.decoder, 'weight') and hasattr(model_eb.decoder, 'weight'):
             model_eb.decoder.weight.data.copy_(model_linear.decoder.weight.data.T.clone())
        else:
            print("Error: SAE decoder structure mismatch for weight copy.")
            return False

    model_linear.eval()
    model_eb.eval()

    test_batch_size = 2
    if is_cc:
        num_enc_l = common_model_args.get("num_encoder_layers") # This is the positional arg for BTKCC
        if num_enc_l is None: # Fallback for direct CrossCoder if API differs, though btkcc_full_args should have it.
             num_enc_l = common_model_args.get("num_layers", NUM_ENCODER_LAYERS_CC) # num_layers is the base CrossCoder positional
        synthetic_input_for_encode = next(generate_synthetic_data(1, test_batch_size, activation_dim, device, num_layers=num_enc_l))
    else:
        synthetic_input_for_encode = next(generate_synthetic_data(1, test_batch_size, activation_dim, device))

    encode_output_eb = model_eb.encode(synthetic_input_for_encode, return_active=True)
    synthetic_features_for_decode_eb = (encode_output_eb[0] if isinstance(encode_output_eb, tuple) else encode_output_eb).detach()

    encode_output_linear = model_linear.encode(synthetic_input_for_encode, return_active=True)
    synthetic_features_for_decode_linear = (encode_output_linear[0] if isinstance(encode_output_linear, tuple) else encode_output_linear).detach()

    features_from_encode_close = torch.allclose(synthetic_features_for_decode_eb, synthetic_features_for_decode_linear, atol=1e-5)
    print(f"  Features from .encode() are close for both models: {features_from_encode_close}")
    if not features_from_encode_close: print(f"    Max diff features_from_encode: {(synthetic_features_for_decode_eb - synthetic_features_for_decode_linear).abs().max()}")

    out_linear_decode = model_linear.decode(synthetic_features_for_decode_linear)
    out_eb_decode = model_eb.decode(synthetic_features_for_decode_eb)

    decode_outputs_close = torch.allclose(out_linear_decode, out_eb_decode, atol=1e-5)
    print(f"  Decode outputs close (using each model's own encoded features): {decode_outputs_close}")
    if not decode_outputs_close: print(f"    Max diff decode: {(out_linear_decode - out_eb_decode).abs().max()}")

    out_linear_full = model_linear(synthetic_input_for_encode)
    out_eb_full = model_eb(synthetic_input_for_encode)
    full_outputs_close = torch.allclose(out_linear_full, out_eb_full, atol=1e-5)
    print(f"  Full model outputs close: {full_outputs_close}")
    if not full_outputs_close: print(f"    Max diff full: {(out_linear_full - out_eb_full).abs().max()}")

    # Training equivalence test
    overall_training_equivalence_passed = True # Start assuming true for this part

    torch.set_grad_enabled(True)
    model_linear.train()
    opt_linear = torch.optim.AdamW(model_linear.parameters(), lr=1e-4)
    opt_linear.zero_grad()
    reconstruction_linear = model_linear(synthetic_input_for_encode) # Use the same input for both
    loss_linear = ((reconstruction_linear - synthetic_input_for_encode)**2).sum()
    loss_linear.backward()
    opt_linear.step()

    model_eb.train()
    opt_eb = torch.optim.AdamW(model_eb.parameters(), lr=1e-4)
    opt_eb.zero_grad()
    reconstruction_eb = model_eb(synthetic_input_for_encode)
    loss_eb = ((reconstruction_eb - synthetic_input_for_encode)**2).sum()

    eb_training_step_attempted = True
    try:
        loss_eb.backward()
        opt_eb.step()
        print("  EB model training step with AdamW succeeded.")
    except RuntimeError as e:
        if "sparse" in str(e).lower():
            print(f"  EB model training step with AdamW failed as expected due to sparse gradients: {e}")
            eb_training_step_attempted = False # Training could not complete equivalently
        else:
            raise # Re-raise unexpected error

    torch.set_grad_enabled(False)
    model_linear.eval()
    model_eb.eval()

    if eb_training_step_attempted:
        training_loss_close = torch.allclose(loss_linear, loss_eb, atol=1e-4)
        print(f"  Training produces similar losses: {training_loss_close}")
        if not training_loss_close: overall_training_equivalence_passed = False

        encoder_weights_close = torch.allclose(model_linear.encoder.weight, model_eb.encoder.weight, atol=1e-4)
        print(f"  Encoder weights close after training: {encoder_weights_close}")
        if not encoder_weights_close: overall_training_equivalence_passed = False; print(f"    Max diff enc W: {(model_linear.encoder.weight - model_eb.encoder.weight).abs().max()}")

        enc_bias_close = True
        if model_linear.encoder.bias is not None and model_eb.encoder.bias is not None:
            enc_bias_close = torch.allclose(model_linear.encoder.bias, model_eb.encoder.bias, atol=1e-4)
        elif model_linear.encoder.bias is not None or model_eb.encoder.bias is not None:
            enc_bias_close = False
        print(f"  Encoder biases close after training: {enc_bias_close}")
        if not enc_bias_close: overall_training_equivalence_passed = False

        dec_bias_close = True
        # Simplified bias check; assumes if one has b_dec, other has it, if one has decoder.bias, other has it.
        if hasattr(model_linear, 'b_dec') and hasattr(model_eb, 'b_dec'):
            dec_bias_close = torch.allclose(model_linear.b_dec, model_eb.b_dec, atol=1e-4)
        elif hasattr(model_linear.decoder, 'bias') and hasattr(model_eb.decoder, 'bias'):
             dec_bias_close = torch.allclose(model_linear.decoder.bias, model_eb.decoder.bias, atol=1e-4)
        elif (hasattr(model_linear, 'b_dec') or hasattr(model_linear.decoder, 'bias')) != \
             (hasattr(model_eb, 'b_dec') or hasattr(model_eb.decoder, 'bias')): # Existence mismatch
            dec_bias_close = False
        print(f"  Decoder biases close after training: {dec_bias_close}")
        if not dec_bias_close: overall_training_equivalence_passed = False

        dec_weights_close = True
        if is_cc:
            if hasattr(model_linear.decoder, 'weight') and hasattr(model_eb.decoder, 'layers'):
                 if model_linear.decoder.weight.shape[0] == len(model_eb.decoder.layers):
                    for i in range(len(model_eb.decoder.layers)):
                        if not torch.allclose(model_linear.decoder.weight.data[i], model_eb.decoder.layers[i].weight.data, atol=1e-3):
                            dec_weights_close = False; break
                 else: dec_weights_close = False
            else: dec_weights_close = False
        else: # SAE
            if hasattr(model_linear.decoder, 'weight') and hasattr(model_eb.decoder, 'weight'):
                dec_weights_close = torch.allclose(model_linear.decoder.weight.data.T, model_eb.decoder.weight.data, atol=1e-3)
            else: dec_weights_close = False
        print(f"  Decoder weights close after training: {dec_weights_close}")
        if not dec_weights_close: overall_training_equivalence_passed = False
    else: # EB training step was not successful
        print("  Post-training weight/loss comparison skipped as EB model training failed.")
        overall_training_equivalence_passed = False # Training was not equivalent

    final_test_result = features_from_encode_close and decode_outputs_close and full_outputs_close and overall_training_equivalence_passed
    print(f"--- End Test {model_name} (Overall Equivalence: {final_test_result}) ---\n")
    return final_test_result

# %%
# --- Main Benchmarking Function ---
def run_benchmarks():
    sae_args = {"activation_dim": ACTIVATION_DIM, "dict_size": DICT_SIZE, "k": K_SAE}
    btkcc_args = {
        "activation_dim": ACTIVATION_DIM,
        "dict_size": DICT_SIZE,
        "num_encoder_layers": NUM_ENCODER_LAYERS_CC,
        "k": K_SAE,
        "encoder_layers_indices": list(range(NUM_ENCODER_LAYERS_CC)),
        "num_decoder_output_layers": NUM_DECODER_LAYERS_CC,
    }

    # Run equivalence tests (optional, can be commented out for speed)
    # test_weight_equivalence(BatchTopKSAE, "BatchTopKSAE", sae_args, ACTIVATION_DIM, DICT_SIZE, DEVICE)
    # test_weight_equivalence(BatchTopKCrossCoder, "BatchTopKCrossCoder", btkcc_args, ACTIVATION_DIM, DICT_SIZE, DEVICE)

    print("\n--- Starting Performance Benchmarks ---")
    model_configs_for_perf = {
        "BatchTopKSAE": (BatchTopKSAE, sae_args, False),
        "BatchTopKCrossCoder": (BatchTopKCrossCoder, btkcc_args, True),
    }
    results: Dict[str, Dict[str, Any]] = {}

    for model_name, (model_class, model_base_args, is_cross_coder_flag) in model_configs_for_perf.items():
        print(f"\n-- Benchmarking {model_name} --")
        results[model_name] = {}

        decoder_configs_to_test = []
        base_decoder_types = ["linear", "embedding_bag"]

        for dec_type in base_decoder_types:
            if dec_type == "embedding_bag":
                decoder_configs_to_test.append({"decoder_type": "embedding_bag", "use_sparse_decoder": True, "descriptive_name": "embedding_bag_sparse_true"})
                decoder_configs_to_test.append({"decoder_type": "embedding_bag", "use_sparse_decoder": False, "descriptive_name": "embedding_bag_sparse_false"})
            else:
                decoder_configs_to_test.append({"decoder_type": "linear", "descriptive_name": "linear"})

        for dec_config in decoder_configs_to_test:
            descriptive_decoder_type = dec_config["descriptive_name"]
            print(f"  Testing Configuration: {descriptive_decoder_type}")

            current_init_args = {**model_base_args, "decoder_type": dec_config["decoder_type"]}
            if "use_sparse_decoder" in dec_config:
                current_init_args["use_sparse_decoder"] = dec_config["use_sparse_decoder"]

            results[model_name][descriptive_decoder_type] = {}

            try:
                model = model_class(**current_init_args).to(DEVICE)
            except Exception as e:
                print(f"Error instantiating {model_name} with {descriptive_decoder_type}: {e}")
                results[model_name][descriptive_decoder_type]["error_instantiation"] = str(e)
                continue

            try:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if not trainable_params:
                    print(f"Warning: No trainable parameters found for {model_name} with {descriptive_decoder_type}. Skipping optimizer init and training.")
                    optimizer = None
                else:
                    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4) # type: ignore[arg-type]
            except Exception as e:
                print(f"Error creating optimizer for {model_name} {descriptive_decoder_type}: {e}")
                results[model_name][descriptive_decoder_type]["error_optimizer"] = str(e)
                continue

            model.train()
            if optimizer: # Proceed with training benchmark only if optimizer was created
                model.train()
                num_data_gen_layers_train = current_init_args.get("num_encoder_layers") if is_cross_coder_flag else None
                train_data_loader = generate_synthetic_data(
                    NUM_STEPS_TRAIN + NUM_STEPS_WARMUP, BATCH_SIZE_TRAIN,
                    current_init_args["activation_dim"], DEVICE,
                    num_layers=num_data_gen_layers_train
                )
                try:
                    for _ in range(NUM_STEPS_WARMUP):
                        x_batch = next(train_data_loader); optimizer.zero_grad()
                        reconstruction = model(x_batch); loss = ((reconstruction - x_batch)**2).sum()
                        loss.backward(); optimizer.step()

                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    start_time_train = time.perf_counter()
                    for _ in range(NUM_STEPS_TRAIN):
                        x_batch = next(train_data_loader); optimizer.zero_grad()
                        reconstruction = model(x_batch); loss = ((reconstruction - x_batch)**2).sum()
                        loss.backward(); optimizer.step()
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    end_time_train = time.perf_counter()
                    avg_time_train = (end_time_train - start_time_train) / NUM_STEPS_TRAIN
                    results[model_name][descriptive_decoder_type]["train_step_time_ms"] = avg_time_train * 1000
                    print(f"    Avg Training Step Time: {avg_time_train*1000:.3f} ms")
                except Exception as e:
                    print(f"Error during training benchmark for {model_name} {descriptive_decoder_type}: {e}")
                    results[model_name][descriptive_decoder_type]["train_step_time_ms"] = "Error: " + str(e)
            else: # No optimizer, skip training benchmark
                print(f"    Skipping training benchmark for {model_name} {descriptive_decoder_type} (no trainable parameters or optimizer error).")
                results[model_name][descriptive_decoder_type]["train_step_time_ms"] = "Skipped - No Optimizer"


            model.eval() # Ensure model is in eval mode for inference benchmarks
            with torch.no_grad():
                num_data_gen_layers_infer = current_init_args.get("num_encoder_layers") if is_cross_coder_flag else None
                # Benchmark for decode-only
                try:
                    # Generate one batch of features by encoding first
                    # This ensures the features_for_decode_batch has the correct structure and device
                    sample_input_for_encode_once = next(generate_synthetic_data(
                        1, BATCH_SIZE_INFER, current_init_args["activation_dim"], DEVICE,
                        num_layers=num_data_gen_layers_infer
                    ))
                    # The model.encode might return tuple, ensure to get the tensor part
                    encoded_output_once = model.encode(sample_input_for_encode_once) # Remove return_active=True if not always used or needed
                    if isinstance(encoded_output_once, tuple): # Handle if encode returns tuple (e.g. with active indices)
                         features_for_decode_batch = encoded_output_once[0].detach()
                    else:
                         features_for_decode_batch = encoded_output_once.detach()

                    for _ in range(NUM_STEPS_WARMUP): model.decode(features_for_decode_batch)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    start_time_infer_decode = time.perf_counter()
                    for _ in range(NUM_STEPS_INFER): model.decode(features_for_decode_batch)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    end_time_infer_decode = time.perf_counter()
                    avg_time_infer_decode = (end_time_infer_decode - start_time_infer_decode) / NUM_STEPS_INFER
                    results[model_name][descriptive_decoder_type]["infer_decode_time_ms"] = avg_time_infer_decode * 1000
                    print(f"    Avg Inference (decode only) Time: {avg_time_infer_decode*1000:.3f} ms")
                except Exception as e:
                    print(f"Error during decode-only inference for {model_name} {descriptive_decoder_type}: {e}")
                    results[model_name][descriptive_decoder_type]["infer_decode_time_ms"] = "Error: " + str(e)

                # Benchmark for full model (encode + decode)
                try:
                    infer_full_data_loader = generate_synthetic_data(
                        NUM_STEPS_INFER + NUM_STEPS_WARMUP, BATCH_SIZE_INFER,
                        current_init_args["activation_dim"], DEVICE,
                        num_layers=num_data_gen_layers_infer
                    )
                    for _ in range(NUM_STEPS_WARMUP): model(next(infer_full_data_loader)) # type: ignore
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    start_time_infer_full = time.perf_counter()
                    for _ in range(NUM_STEPS_INFER): model(next(infer_full_data_loader)) # type: ignore
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    end_time_infer_full = time.perf_counter()
                    avg_time_infer_full = (end_time_infer_full - start_time_infer_full) / NUM_STEPS_INFER
                    results[model_name][descriptive_decoder_type]["infer_full_model_time_ms"] = avg_time_infer_full * 1000
                    print(f"    Avg Inference (encode-decode) Time: {avg_time_infer_full*1000:.3f} ms")
                except Exception as e:
                    print(f"Error during full inference for {model_name} {descriptive_decoder_type}: {e}")
                    results[model_name][descriptive_decoder_type]["infer_full_model_time_ms"] = "Error: " + str(e)


    print("\n--- Benchmark Summary ---")
    for model_name_sum, model_results in results.items():
        print(f"  {model_name_sum}:")
        for dec_type_sum, times in model_results.items():
            print(f"    {dec_type_sum}:")
            if isinstance(times, dict):
                for metric, t_val in times.items():
                    if isinstance(t_val, float): print(f"      {metric}: {t_val:.3f} ms")
                    else: print(f"      {metric}: {t_val}")
            else: print(f"      Error: {times}")
    print("--- Benchmarking Script Complete ---")

# %%
# --- Run Benchmarks ---
if __name__ == "__main__":
    # Store results globally for plotting if needed, or pass around
    benchmark_results_global = run_benchmarks()

# %%
# --- Plotting Results ---
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if 'benchmark_results_global' in locals() and benchmark_results_global:
        print("\n--- Generating Benchmark Plots ---")
        results_to_plot = benchmark_results_global # Use the global variable

        model_names = list(results_to_plot.keys())

        # Define the configurations we expect to plot, corresponds to descriptive_decoder_type
        decoder_configs_plot_order = ["linear", "embedding_bag_sparse_true", "embedding_bag_sparse_false"]

        metrics_to_plot = [
            ("train_step_time_ms", "Avg Training Step Time (ms)"),
            ("infer_decode_time_ms", "Avg Inference (Decode Only) Time (ms)"),
            ("infer_full_model_time_ms", "Avg Inference (Encode-Decode) Time (ms)")
        ]

        for model_name_plt in model_names:
            print(f"  Plotting for model: {model_name_plt}")
            model_data = results_to_plot.get(model_name_plt, {})

            num_metrics = len(metrics_to_plot)
            num_configs = len(decoder_configs_plot_order)

            bar_width = 0.25
            fig_height = 5 * num_metrics
            # Try to create a subplot for each metric for a given model
            # Or, create separate figures for each metric for clarity

            for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
                plt.figure(figsize=(10, 6)) # Create a new figure for each metric

                values_for_metric = []
                actual_configs_plotted = []

                for config_name in decoder_configs_plot_order:
                    config_data = model_data.get(config_name, {})
                    value = config_data.get(metric_key)

                    if isinstance(value, (float, int)):
                        values_for_metric.append(value)
                        actual_configs_plotted.append(config_name)
                    elif isinstance(value, str) and "Error" in value:
                        print(f"    Warning: Error recorded for {model_name_plt}, {config_name}, {metric_key}: {value}. Plotting as 0.")
                        values_for_metric.append(0) # Plot errors as 0
                        actual_configs_plotted.append(config_name + " (Error)")
                    elif isinstance(value, str) and "Skipped" in value:
                        print(f"    Info: Skipped data for {model_name_plt}, {config_name}, {metric_key}: {value}. Plotting as 0.")
                        values_for_metric.append(0) # Plot skipped as 0
                        actual_configs_plotted.append(config_name + " (Skipped)")
                    else:
                        # Handle cases where a config might be missing entirely for a metric (e.g. if instantiation failed)
                        print(f"    Warning: Missing data for {model_name_plt}, {config_name}, {metric_key}. Plotting as 0.")
                        values_for_metric.append(0)
                        actual_configs_plotted.append(config_name + " (Missing)")


                x = np.arange(len(actual_configs_plotted))

                plt.bar(x, values_for_metric, bar_width, label=metric_label)

                plt.ylabel('Time (ms)')
                plt.title(f'{model_name_plt} - {metric_label}')
                plt.xticks(x, actual_configs_plotted, rotation=15, ha="right")
                plt.legend()
                plt.tight_layout()
                plt.show()
        print("--- Plotting Complete ---")
    else:
        print("\nNo benchmark results to plot (or script not run as main).")
