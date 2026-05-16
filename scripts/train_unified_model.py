"""
train_unified_model.py
===========================================================
End-to-end training script for the Unified MultiTask Model.

Usage:
    python scripts/train_unified_model.py [--epochs 100] [--lr 0.001] [--batch 16]

Steps:
    1. Load unified_training_data.csv
    2. Instantiate ESM2HybridEncoder + UnifiedMultiTaskModel
    3. Split into train/val (80/20)
    4. Train with UnifiedTrainer (masked loss + early stopping)
    5. Save model to models/unified_multitask_best.pt
"""

import argparse
import os
import sys

# Fix macOS OpenMP duplicate library crash (PyTorch + NumPy/SciPy conflict)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add src/ to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
from torch.utils.data import DataLoader, random_split

from esm2_hybrid_encoder import ESM2HybridEncoder
from unified_multitask_model import UnifiedMultiTaskModel, UNIFIED_TASKS
from unified_dataset import UnifiedAntibodyDataset, unified_collate_fn
from unified_trainer import UnifiedTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Unified MultiTask Model")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=256, help="Encoder hidden dim")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default=os.path.join(PROJECT_ROOT, "models"))
    parser.add_argument("--cache", default=os.path.join(PROJECT_ROOT, "data", "esm2_embeddings_cache.pt"),
                        help="Path to pre-computed ESM-2 embeddings (.pt). Use 'none' to disable.")
    args = parser.parse_args()

    # Resolve cache path
    if args.cache.lower() == "none" or not os.path.exists(args.cache):
        args.cache = None
        print("  ESM-2 cache: NOT FOUND — will compute embeddings live (slow!)")
    else:
        print(f"  ESM-2 cache: {args.cache}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load dataset
    print(f"Loading data from {args.data}")
    dataset = UnifiedAntibodyDataset(args.data, compute_biophys=True,
                                      embedding_cache_path=args.cache)
    print(f"  Total samples: {len(dataset)}")

    # 2. Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"  Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=unified_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=unified_collate_fn,
    )

    # 3. Model
    encoder = ESM2HybridEncoder(hidden_dim=args.hidden)
    model = UnifiedMultiTaskModel(
        encoder=encoder,
        encoder_dim=args.hidden,
        tasks=UNIFIED_TASKS,
    )
    print(f"\nModel: {model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 5. Train
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        patience=args.patience,
        save_dir=args.save_dir,
    )

    print(f"\nStarting training: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
    summary = trainer.train(epochs=args.epochs, verbose=True)

    # 6. Report
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"  Epochs trained: {summary['epochs_trained']}")
    print(f"  Best val loss:  {summary['best_val_loss']:.6f}")
    print(f"  Final train:    {summary['final_train_loss']:.6f}")
    print(f"  Final val:      {summary['final_val_loss']:.6f}")
    print(f"  Time elapsed:   {summary['elapsed_seconds']:.1f}s")
    print(f"  Model saved:    {summary['model_path_best']}")


if __name__ == "__main__":
    main()
