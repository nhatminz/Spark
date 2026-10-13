# Spark

Standalone draft pretraining and GRPO policy training with two methods:

- `tree`: a one-layer draft predicts hidden states and token distributions. The proposal tree expands according to cumulative probabilities, retains high-confidence nodes, and is verified with tree attention by following tokens sampled by the target. The draft objective combines hidden-state Smooth L1 loss and soft cross entropy.
- `block`: a three-layer draft predicts MASK positions in parallel. Each block attends only to the history before its anchor and positions within that block. It shares the target's embedding and output head and uses soft cross entropy with weights that decrease by position. Greedy proposals are verified through rejection sampling, followed by a recovery token or a target bonus token.

Add `--correction` during training to enable online correction. A persistent projector is learned across batches; the correction matrix is reset at the start of each rollout batch and updated using contexts actually visited during verification. Updates use only the union of the two distributions' top-k coordinates. Correction is disabled by default.

There is no application logging, progress bar, background service, or dependency on other projects in the workspace. Libraries may still emit warnings or errors. Checkpoints are the program's only output artifacts.

## Project structure

```text
Spark/
├── pretrain.py              # Draft pretraining loop
├── train.py                 # GRPO and online draft training loop
├── helper/
│   ├── __init__.py
│   ├── arguments.py         # CLI arguments and configuration validation
│   ├── data.py              # Data loading, chat templates, and batching
│   ├── runtime.py           # Target loading, precision, and context features
│   ├── objectives.py        # Policy/draft objectives and draft updates
│   ├── optimizers.py        # FP32 master weights and gradient accumulation
│   ├── generation.py        # Tree/block generation and verification
│   ├── correction.py        # Optional online correction
│   ├── rewards.py           # Mathematical answer and format rewards
│   └── checkpoints.py       # Checkpoint and RNG state saving/loading
├── models/
│   ├── __init__.py
│   ├── draft.py             # Tree/block draft architectures
│   └── policy.py            # Policy adapters and weight restoration
├── requirements.txt
└── README.md
```

`pretrain.py` and `train.py` orchestrate the training loops. Modules in `helper/` and `models/` contain the implementations required by these two entry points. This organization preserves the CLI commands and checkpoint format.

## Environment and inputs

Use Python 3.10 or later and a CUDA-enabled PyTorch build compatible with your GPU. You can use an existing environment if it already satisfies `requirements.txt`. Otherwise, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Each command runs one process on one GPU. Select the GPU with `CUDA_VISIBLE_DEVICES`; do not use `torchrun`. The implementation uses PyTorch SDPA and requires no custom CUDA extension compilation. The model and tokenizer must already be available locally; the program does not download them.

The target must use a dense decoder, RMSNorm, gated SiLU, grouped-query attention, and standard RoPE. Sliding-window attention, MoE, and scaled RoPE are outside the supported scope. The tokenizer must provide an EOS token and a chat template that preserves token prefixes when conversation turns are appended. For `block`, the embedding must contain a row outside the tokenizer's vocabulary to use as MASK. The program finds this row automatically, or you can specify it with `--mask-id`.

Set these three environment variables to your own input locations before running:

- `MODEL`: the target model and tokenizer directory.
- `PRETRAIN_DATA`: the conversation dataset file.
- `TRAIN_DATA`: the question-and-answer dataset file.

Datasets may use JSON (an array of objects), JSONL, or Parquet. Pretraining schema:

```json
[
  {"messages": [
    {"role": "user", "content": "What is 1 + 1?"},
    {"role": "assistant", "content": "2"}
  ]}
]
```

The alternative `conversations` schema with `from`/`value` fields is also accepted. Only assistant turns contribute to the loss; tokens truncated by `--max-length` are excluded.

Policy training schema:

```json
[
  {"question": "What is 1 + 1?", "answer": "2"}
]
```

You can replace `question` with `prompt` (a string or a list of messages), or use `messages`; `reward_model.ground_truth` can replace `answer`. A `question` is wrapped in mathematical reasoning and answer instructions, while `prompt`/`messages` use the content you provide. Prompts exceeding the length limit are rejected. The reward is mathematical answer correctness plus `0.2` if the response contains exactly one `\n</think>\n` segment. Unparseable ground truth raises an error before training.

## Pretraining

Run from the `Spark` directory:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
  --method tree --model "$MODEL" --data "$PRETRAIN_DATA" \
  --output checkpoints/tree_pretrain

CUDA_VISIBLE_DEVICES=1 python pretrain.py \
  --method block --model "$MODEL" --data "$PRETRAIN_DATA" \
  --output checkpoints/block_pretrain
```

Defaults: a frozen target, 5 epochs, learning rate `5e-5`, batch size `8`, gradient accumulation `4`, and a `2048`-token length limit. For `block`, the block size is `16`, with `16` anchors per sequence. Override these settings with `--layers`, `--block-size`, `--anchors`, `--max-length`, or `--lr`.

## Policy training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --method tree --model "$MODEL" --data "$TRAIN_DATA" \
  --pretrained checkpoints/tree_pretrain/latest.pt \
  --output checkpoints/tree_train

CUDA_VISIBLE_DEVICES=1 python train.py \
  --method block --model "$MODEL" --data "$TRAIN_DATA" \
  --pretrained checkpoints/block_pretrain/latest.pt \
  --output checkpoints/block_train
```

To enable correction, add `--correction` and choose a separate `--output` directory. Each method's pretrained checkpoint can initialize runs with or without correction.

Defaults: 1 epoch, `8` prompts per batch, `8` responses per prompt, gradient accumulation `4`, target learning rate `1e-6`, draft learning rate `1e-4`, temperature `1.0`, top-p `0.95`, and KL coefficient `0.04`. Prompt and response limits are both `2048` tokens. GRPO normalizes rewards within each response group and averages the loss by sequence. Groups with identical rewards do not update the target but still train the draft.

The default `--policy-update native` uses a rank-`64` adapter with alpha `32` for `tree` and updates the full target for `block`. Select `--policy-update adapter` or `--policy-update full` explicitly to use the same update scope for both methods. Full training keeps a frozen reference model and requires enough VRAM for the policy, reference, FP32 master weights, and optimizer. Move the reference to CPU with `--reference-device cpu` if needed, at the cost of lower speed.

## Iterations and checkpoints

Accumulation performs **actual gradient accumulation**. With the default settings, an iteration contains up to 4 batches: 32 conversations during pretraining, or 32 prompts / 256 responses during policy training. The target updates at most once per iteration; the draft updates once per rollout batch during policy training. Forward and backward passes process sequences individually to reduce memory usage, so `--batch-size` specifies the logical batch size. The final batch or iteration may be smaller.

Checkpoints are saved every `50` iterations and at completion by default; change the interval with `--save-every`. Each `step_N.pt` contains the draft, policy when applicable, optimizer state, data position, and RNG state. `latest.pt` points to the newest checkpoint. Input paths are not stored in checkpoints. Both entry points read checkpoints produced by this implementation and do not directly load other backends' checkpoint formats.

To resume, use the corresponding command with the same settings and add:

```bash
--resume checkpoints/tree_train/latest.pt
```

For pretraining, point to a pretraining checkpoint instead. `--pretrained` is not required when resuming policy training. Keep the target/tokenizer, dataset, batch size, accumulation, epoch count, and algorithm settings unchanged; the program checks configuration and dataset checksums. Use `--max-steps N` to stop after iteration N and save a checkpoint for resuming later.

This is an independent implementation of the algorithms. Batching, sampling order, and gradient accumulation follow the definitions above; runtime and stochastic trajectories should not be assumed equivalent to other backends. Validation used small target models with PyTorch 2.4.1 and Transformers 4.57.1. Validate your actual model configuration before starting a long training run.
