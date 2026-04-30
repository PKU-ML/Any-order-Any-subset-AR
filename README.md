# Any-order Any-subset Autoregressive Modeling (A3)

This repository implements [Any-order Any-subset Autoregressive Modeling (A3)](https://arxiv.org/abs/2601.13228) on top of the LLaMA architecture. The A3 framework extends the standard autoregressive (AR) factorization to arbitrary token groups and generation orders, preserving the multi-layer dependency modeling of AR while inheriting the flexibility of diffusion models for parallel and bidirectional generation. 

The core implementation uses a **two-stream attention mechanism** inspired by XLNet, where the generation process is reformulated into predicting subsets of tokens sequentially. 

## File Structure
- `llama_xlnet.py`: Contains the core modeling classes (`LlamaTwoStreamModel`, `LlamaTwoStreamForCausalLM`, `LlamaTwoStreamAttention`).
- `example_usage.py`: A training script to train the A3 model on an example dataset with different grouping strategies.

## Usage Examples

### Initialization and Loading
You can initialize an A3 model from a pretrained standard LLaMA checkpoint. The model will handle common key name mapping and load the weights correctly.

```python
from llama_xlnet import LlamaTwoStreamForCausalLM

model = LlamaTwoStreamForCausalLM.from_llama_checkpoint("meta-llama/Llama-3.1-8B")
```

### Forward Pass with Custom Grouping

```python
import torch

input_ids = torch.tensor([[101, 2054, 304, 502, 608, 705]])
# Custom grouping: tokens 0,1 in group 0; tokens 2,3 in group 1; tokens 4,5 in group 2
# Note: groups don't have to be monotonic! 
group_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])

outputs = model(
    input_ids=input_ids,
    group_ids=group_ids,
    attention_mask=attention_mask,
    labels=input_ids
)

print(outputs.loss)
```

### Inference: Group-by-Group Generation
Generate sequence by generating groups incrementally:

```python
# Assuming a prefix is provided in group 0, and we want to generate group 1 and 2
group_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
generated_ids = model.generate_two_stream(
    input_ids=input_ids,
    group_ids=group_ids,
    max_length=6,
    temperature=1.0,
    do_sample=True
)
```

### Inference: Dynamic Resampling
Alternatively, use the dynamic resampling strategy (Algorithm 2) which dynamically evaluates and commits a specific number of tokens at each step based on the model's confidence, entropy, or randomly:

```python
# Assuming prefix is provided as non-pad tokens in input_ids
# pad_token_id represents blank positions to be filled
generated_ids = model.generate_dynamic_resampling(
    input_ids=input_ids,
    max_length=128,
    step_size=4,             # Commit 4 tokens per step
    criterion="confidence",  # Options: "confidence", "entropy", "random"
    temperature=1.0,
    do_sample=True
)
```

## Training

The script `example_usage.py` provides an example of adapting a LLaMA model on the FineWeb dataset using A3's progressive token grouping strategies.

```bash
torchrun --nproc_per_node=8 example_usage.py \
  --dataset_name HuggingFaceFW/fineweb \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --output_dir ./a3_fineweb_checkpoint \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 2048 \
  --grouping_strategy 3 \
  --bf16
```

### Supported Grouping Strategies (Curriculum Stages)
- `0`: Standard AR initialization (1 token per group).
- `1`: Group expansion with size 2.
- `2`: Group expansion with size 4.
- `3`: Order permutation with groups of size 4 (Any-order).
- `4`: Order permutation with groups of size 8 (Any-order).


## Requirements
- PyTorch >= 2.0
- Transformers
- Datasets
- (Optional) DeepSpeed
