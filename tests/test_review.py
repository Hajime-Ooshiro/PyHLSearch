#!/usr/bin/env python
"""Quick test to understand the progress update logic."""

import sys
sys.path.insert(0, '.')

# Check the progress update logic more carefully
# The question is: should pbar.update(1) be called for every node,
# or only every postfix_update_interval nodes?

# In tqdm:
# - pbar.update(n) increments the counter by n
# - mininterval controls how often the display is refreshed
# - pbar.set_postfix() updates the postfix metadata

# The original code called pbar.update(1) for every node, meaning every node incremented the counter
# but the display was only refreshed every mininterval seconds

# The new code only calls pbar.update(1) every postfix_update_interval nodes
# This means if postfix_update_interval=10000, the bar will jump by 1 every 10000 nodes processed

print("Checking the change...")
print("OLD: pbar.update(1) called for EVERY node")
print("NEW: pbar.update(1) called only when node_count % postfix_update_interval == 0")
print()
print("This means with postfix_update_interval=10000:")
print("- OLD: counter increments by 1 for each of 10000 nodes")
print("- NEW: counter increments by 1 only once per 10000 nodes")
print()
print("This is likely a performance bug - the progress bar appears frozen")
