#!/usr/bin/env python3

import os
import sys

fairseq_root = "/workspace/fongbe_unsupervised/fairseq"
sys.path.insert(0, fairseq_root)
os.environ['PYTHONPATH'] = fairseq_root

from fairseq import checkpoint_utils, utils

def debug_config():
    checkpoint_path = "/workspace/fongbe_unsupervised/fongbe_checkpoints_official/checkpoint_last.pt"
    user_dir = "/workspace/fongbe_unsupervised/fairseq/examples/wav2vec/unsupervised"
    
    utils.import_user_module(user_dir)
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
    cfg = state['cfg']
    
    print("Model config:")
    print(cfg.model)
    
    print(f"\nSegmentation config:")
    print(cfg.model.segmentation)

if __name__ == "__main__":
    debug_config()