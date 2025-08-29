#!/usr/bin/env python3
"""
Start wav2vec-U 2.0 training for Fongbe ASR
"""

import os
import sys
import subprocess
from pathlib import Path

def create_training_config(features_dir, manifests_dir, config_path):
    """Create training configuration file"""

    config_content = f"""# Fongbe wav2vec-U 2.0 Training Configuration
# Based on the official w2vu2.yaml config

common:
  fp16: false
  fp16_no_flatten_grads: true
  log_format: json
  log_interval: 50
  tensorboard_logdir: /workspace/fongbe_tensorboard
  reset_logging: false
  suppress_crashes: false

checkpoint:
  save_interval: 500
  save_interval_updates: 500
  no_epoch_checkpoints: true
  best_checkpoint_metric: weighted_lm_ppl
  save_dir: /workspace/fongbe_checkpoints

distributed_training:
  distributed_world_size: 1

task:
  _name: unpaired_audio_text
  data: /workspace/{features_dir}  # Directory containing features
  text_data: /workspace/fongbe_data/manifests  # Directory for text manifests
  labels: phn
  sort_by_length: false
  unfiltered: false
  max_length: null
  append_eos: false
  kenlm_path: null
  aux_target_postfix: km

dataset:
  num_workers: 2
  batch_size: 1   # Single sample per batch for small dataset
  skip_invalid_size_inputs_valid_test: true
  disable_validation: true
  validate_interval: 1000000
  validate_interval_updates: 1000000

criterion:
  _name: model
  log_keys:
    - accuracy_dense
    - accuracy_token
    - temp
    - code_ppl

optimization:
  max_update: 1000  # Short training for testing
  clip_norm: 5.0

optimizer:
  _name: composite
  groups:
    generator:
      optimizer:
        _name: adam
        adam_betas: [0.5,0.98]
        adam_eps: 1e-06
        weight_decay: 0
        lr: [0.00005]
      lr_scheduler:
        _name: fixed
        warmup_updates: 0
        lr: [0.00005]
    discriminator:
      optimizer:
        _name: adam
        adam_betas: [0.5,0.98]
        adam_eps: 1e-06
        weight_decay: 0.0001
        lr: [0.0003]
      lr_scheduler:
        _name: fixed
        warmup_updates: 0
        lr: [0.0003]

lr_scheduler: pass_through

model:
  _name: wav2vec_u

  # Discriminator settings
  discriminator_dim: 384
  discriminator_depth: 2
  discriminator_kernel: 8
  discriminator_linear_emb: false
  discriminator_causal: true
  discriminator_max_pool: false
  discriminator_act_after_linear: false
  discriminator_dropout: 0.0
  discriminator_weight_norm: false

  # Generator settings (wav2vec-U 2.0 style)
  generator_stride: 3    # 50Hz -> 16Hz output frequency
  generator_kernel: 9
  generator_bias: false
  generator_dropout: 0.1
  generator_batch_norm: 30  # Critical for wav2vec-U 2.0
  generator_residual: true

  # Loss weights
  smoothness_weight: 1.5
  smoothing: 0
  smoothing_one_sided: false
  gumbel: false
  hard_gumbel: false
  gradient_penalty: 1.0
  code_penalty: 3.0
  temp: [2, 0.1, 0.99995]
  input_dim: 1024  # wav2vec 2.0 feature dimension
  mmi_weight: 0.5  # Auxiliary loss weight
  target_dim: 64   # Pseudo-label dimension

  segmentation:
    type: JOIN
    mean_pool_join: false
    remove_zeros: false

hydra:
  job:
    config:
      override_dirname:
        kv_sep: ':'
        item_sep: '__'
        exclude_keys:
          - run_config
          - distributed_training.distributed_port
          - common.user_dir
          - task.data
          - task.kenlm_path
          - task.text_data
          - model.generator_layers
          - task.labels
          - task.force_model_seed
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_fongbe_wav2vec_u2.py <features_dir>")
        print("Example: python train_fongbe_wav2vec_u2.py fongbe_features")
        sys.exit(1)

    features_dir = Path(sys.argv[1])

    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        sys.exit(1)

    # Check required files
    required_files = ["train.npy", "train.lengths", "train.tsv", "dict.phn.txt"]
    for req_file in required_files:
        if not (features_dir / req_file).exists():
            print(f"❌ Required file not found: {features_dir / req_file}")
            sys.exit(1)

    manifests_dir = features_dir.parent / "fongbe_data" / "manifests"
    config_path = Path("/workspace/fongbe_training_config.yaml")

    print(f"🔧 Creating training configuration...")
    create_training_config(features_dir, manifests_dir, config_path)
    print(f"   → {config_path}")

    # Create checkpoint directory
    os.makedirs("/workspace/fongbe_checkpoints", exist_ok=True)

    # Set up training command
    cmd = [
        "python", "fairseq/fairseq_cli/hydra_train.py",
        "--config-path", "/workspace",
        "--config-name", "fongbe_training_config.yaml",
        "common.user_dir=/workspace/fairseq/examples/wav2vec/unsupervised"
    ]

    print(f"🚀 Starting wav2vec-U 2.0 training...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Features: {features_dir}")
    print(f"   Logs: /workspace/fongbe_tensorboard")
    print(f"   Checkpoints: /workspace/fongbe_checkpoints")

    # Run training
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = "/workspace/fairseq:" + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(cmd, env=env, cwd="/workspace")
        if result.returncode == 0:
            print(f"✅ Training completed successfully!")
        else:
            print(f"❌ Training failed with exit code {result.returncode}")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
