"""
Master Benchmark Runner and Acceptance Verification.
Executes both MulT-Lite on MELD and Early-Fusion on CORAL, producing comparison metrics.
"""

import os
import sys
import json
from run_meld_benchmark import run_meld_benchmark
from run_coral_ablation import run_coral_ablation

def main():
    print("="*75)
    print(" RUNNING ALL BENCHMARKS & ARCHITECTURAL ABLATIONS")
    print("="*75)

    print("\n[Step 1/2] Running MulT-Lite on MELD (Neurotypical Benchmark)...")
    meld_f1, meld_acc = run_meld_benchmark()

    print("\n[Step 2/2] Running Early-Fusion Baseline on CORAL (Dysarthric Ablation)...")
    coral_f1, coral_acc = run_coral_ablation()

    print("\n" + "="*75)
    print(" BENCHMARK COMPARISON SUMMARY")
    print("="*75)
    print(f"{'Experiment':<35} | {'Weighted F1':<15} | {'Accuracy':<15}")
    print("-" * 75)
    print(f"{'MELD: MulT-Lite (Cross-Attention)':<35} | {meld_f1:<15.4f} | {meld_acc:<15.4f}")
    print(f"{'CORAL: BiLSTM-Attention (MELD Baseline)':<35} | {coral_f1:<15.4f} | {coral_acc:<15.4f}")
    print("="*75)

if __name__ == "__main__":
    main()
