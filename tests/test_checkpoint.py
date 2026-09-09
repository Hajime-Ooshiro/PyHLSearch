#!/usr/bin/env python
"""Verify checkpoint serialization bug."""

import json
import sys
import tempfile
sys.path.insert(0, '.')

import HLSearch as hl
import numpy as np

config = hl.SearchConfig(primes=[2,3], depth=2, target=10, max_depth=2, cols=8)
shift_table = hl.build_shift_table([2,3], 8)

with tempfile.TemporaryDirectory() as tmp:
    chk_path = f"{tmp}/test.json"
    state = hl.State(config, shift_table, checkpoint_path=chk_path)
    
    # Set up some state
    state.key = [0, 1]
    state.zero_mask = np.array([0b01010101], dtype=np.uint64)
    state.max_count = 7
    state.results = 2
    state.target_results = 1
    state.shifts = [[0, 1], [1, 0]]
    state.target_shifts = [[0, 1]]
    state.max_shifts = [[1, 0]]
    state.node_count = 42
    state._stack = [
        [0, state.zero_mask.copy(), 1, 2],
        [1, np.array([0b00110011], dtype=np.uint64), 0, 3],
    ]
    
    # Save checkpoint
    state._save_checkpoint()
    
    # Check JSON content
    with open(chk_path, 'r') as f:
        saved = json.load(f)
    
    print("Checkpoint keys:", list(saved.keys()))
    print("Stack in checkpoint:", len(saved.get('stack', [])), "frames")
    
    # Load and verify
    restored = hl.State(config, shift_table)
    restored._load_checkpoint(chk_path)
    
    print("Restored key:", restored.key)
    print("Restored max_count:", restored.max_count)
    print("Restored _stack length:", len(restored._stack))
    print("Success!")
