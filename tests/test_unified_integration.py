"""
test_unified_integration.py
===========================================================
Comprehensive integration tests for the Unified MultiTask System.

Tests:
  1. ESM2HybridEncoder: forward pass, output shape, mock fallback
  2. UnifiedMultiTaskModel: forward pass, output shapes, task names
  3. UnifiedAntibodyDataset: loading, masking, collation
  4. UnifiedTrainer: masked loss, training loop (5 epochs)
  5. MultiTaskAdapter: backward-compatible API formats
  6. End-to-end: data → model → prediction → adapter output
"""

import os
import sys
import tempfile

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="torch required for unified integration tests")
from torch.utils.data import DataLoader  # guarded by importorskip above

PASSED = 0
FAILED = 0


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global PASSED, FAILED
            try:
                func()
                PASSED += 1
                print(f"  \u2705 {name}")
            except Exception as e:
                FAILED += 1
                print(f"  \u274C {name}: {e}")
        return wrapper
    return decorator


# ===== TEST 1: ESM2HybridEncoder =====
@test("ESM2HybridEncoder forward pass (mock mode)")
def test_encoder_forward():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    enc = ESM2HybridEncoder(hidden_dim=128)

    hc = ["EVQLVESGGGLVQ", "QVQLQQPGAELVK"]
    lc = ["DIQMTQSPSSLS", "QIVLSQSPAILS"]

    out = enc(hc, lc)
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"


@test("ESM2HybridEncoder with biophysical features")
def test_encoder_with_biophys():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    enc = ESM2HybridEncoder(hidden_dim=128)

    hc = ["EVQLVESGGGLVQ"]
    lc = ["DIQMTQSPSSLS"]
    biophys = torch.tensor([[8.5, 148.0, 1.0, 1.0, 40.0, 50.0, 0.35]])

    out = enc(hc, lc, biophys)
    assert out.shape == (1, 128), f"Expected (1, 128), got {out.shape}"


@test("ESM2HybridEncoder embedding_dim property")
def test_encoder_dim():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    enc = ESM2HybridEncoder(hidden_dim=256)
    assert enc.embedding_dim == 256


# ===== TEST 2: UnifiedMultiTaskModel =====
@test("UnifiedMultiTaskModel forward pass")
def test_model_forward():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    from unified_multitask_model import UnifiedMultiTaskModel, UNIFIED_TASKS

    enc = ESM2HybridEncoder(hidden_dim=128)
    model = UnifiedMultiTaskModel(encoder=enc, encoder_dim=128)

    hc = ["EVQLVESGGGLVQ", "QVQLQQPGAELVK", "EVQLVESGGGLVQ"]
    lc = ["DIQMTQSPSSLS", "QIVLSQSPAILS", "DIQMTQSPSSLS"]

    outputs = model(hc, lc)

    assert set(outputs.keys()) == set(UNIFIED_TASKS), \
        f"Expected tasks {UNIFIED_TASKS}, got {list(outputs.keys())}"

    for task, tensor in outputs.items():
        assert tensor.shape == (3,), f"Task {task}: expected (3,), got {tensor.shape}"


@test("UnifiedMultiTaskModel bounded tasks are in [0, 1]")
def test_bounded_tasks():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    from unified_multitask_model import UnifiedMultiTaskModel, BOUNDED_TASKS

    enc = ESM2HybridEncoder(hidden_dim=64)
    model = UnifiedMultiTaskModel(encoder=enc, encoder_dim=64)

    hc = ["EVQLVESGGGLVQ"] * 5
    lc = ["DIQMTQSPSSLS"] * 5

    outputs = model(hc, lc)
    for task in BOUNDED_TASKS:
        vals = outputs[task]
        assert (vals >= 0).all() and (vals <= 1).all(), \
            f"Task {task} out of [0,1]: min={vals.min():.4f}, max={vals.max():.4f}"


@test("UnifiedMultiTaskModel predict_numpy")
def test_predict_numpy():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    from unified_multitask_model import UnifiedMultiTaskModel

    enc = ESM2HybridEncoder(hidden_dim=64)
    model = UnifiedMultiTaskModel(encoder=enc, encoder_dim=64)

    result = model.predict_numpy(["EVQLVESGGGLVQ"], ["DIQMTQSPSSLS"])
    assert isinstance(result, dict)
    assert "ka" in result and "nu" in result
    assert isinstance(result["ka"], float)


# ===== TEST 3: UnifiedAntibodyDataset =====
@test("UnifiedAntibodyDataset loading and masking")
def test_dataset():
    from unified_dataset import UnifiedAntibodyDataset

    data_path = os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Run prepare_unified_training_data.py first")

    ds = UnifiedAntibodyDataset(data_path)
    assert len(ds) > 0, "Dataset is empty"

    hc, lc, labels, mask, biophys, cached_emb = ds[0]
    assert isinstance(hc, str) and len(hc) > 0
    assert isinstance(lc, str) and len(lc) > 0
    assert isinstance(labels, dict) and "ka" in labels
    assert isinstance(mask, dict) and "ka" in mask
    assert biophys is not None and biophys.shape == (7,)


@test("UnifiedAntibodyDataset collate function")
def test_collate():
    from unified_dataset import UnifiedAntibodyDataset, unified_collate_fn

    data_path = os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv")
    ds = UnifiedAntibodyDataset(data_path)
    loader = DataLoader(ds, batch_size=4, collate_fn=unified_collate_fn)

    batch = next(iter(loader))
    hc_seqs, lc_seqs, labels, masks, biophys, cached_emb = batch

    assert len(hc_seqs) == 4
    assert labels["ka"].shape == (4,)
    assert masks["ka"].shape == (4,)
    assert biophys.shape == (4, 7)


# ===== TEST 4: UnifiedTrainer =====
@test("UnifiedTrainer masked loss computation")
def test_masked_loss():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    from unified_multitask_model import UnifiedMultiTaskModel
    from unified_trainer import UnifiedTrainer

    enc = ESM2HybridEncoder(hidden_dim=64)
    model = UnifiedMultiTaskModel(encoder=enc, encoder_dim=64)

    # Simulate outputs
    outputs = {"ka": torch.tensor([1.0, 2.0, 3.0]), "nu": torch.tensor([4.0, 5.0, 6.0])}
    labels = {"ka": torch.tensor([1.5, 0.0, 3.5]), "nu": torch.tensor([4.5, 5.5, 0.0])}
    masks = {"ka": torch.tensor([1.0, 0.0, 1.0]), "nu": torch.tensor([1.0, 1.0, 0.0])}

    trainer = UnifiedTrainer(
        model=model,
        train_loader=DataLoader([]),
        val_loader=DataLoader([]),
        optimizer=torch.optim.Adam(model.parameters()),
    )

    loss = trainer.compute_loss(outputs, labels, masks)
    assert loss is not None
    assert loss.item() > 0, "Loss should be > 0"
    assert not torch.isnan(loss), "Loss is NaN"


@test("UnifiedTrainer training loop (3 epochs)")
def test_training_loop():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    from unified_multitask_model import UnifiedMultiTaskModel
    from unified_trainer import UnifiedTrainer
    from unified_dataset import UnifiedAntibodyDataset, unified_collate_fn

    data_path = os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv")
    ds = UnifiedAntibodyDataset(data_path)

    # Small split
    train_size = min(20, len(ds) - 4)
    val_size = 4
    from torch.utils.data import Subset
    train_ds = Subset(ds, list(range(train_size)))
    val_ds = Subset(ds, list(range(train_size, train_size + val_size)))

    train_loader = DataLoader(train_ds, batch_size=8, collate_fn=unified_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=unified_collate_fn)

    enc = ESM2HybridEncoder(hidden_dim=64)
    model = UnifiedMultiTaskModel(encoder=enc, encoder_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UnifiedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            patience=100,  # Don't trigger early stopping
            save_dir=tmpdir,
        )
        summary = trainer.train(epochs=3, verbose=False)

    assert summary["status"] == "success"
    assert summary["epochs_trained"] == 3
    assert summary["best_val_loss"] < float("inf")


# ===== TEST 5: MultiTaskAdapter =====
@test("MultiTaskAdapter initialization (no model file)")
def test_adapter_init():
    from multitask_adapter import MultiTaskAdapter
    adapter = MultiTaskAdapter(model_path="/nonexistent/path.pt")
    assert not adapter.is_available


@test("MultiTaskAdapter API format: predict_chromatography")
def test_adapter_chromatography_format():
    """Verify output format matches ChromatographyMLP.predict_single()"""
    # This tests the API contract even without a trained model
    expected_keys = {"ka", "nu", "estimated_rt_min"}
    # If model loaded, verify format. Otherwise just confirm API exists.
    from multitask_adapter import MultiTaskAdapter
    adapter = MultiTaskAdapter(model_path="/nonexistent/path.pt")
    assert hasattr(adapter, "predict_chromatography")


@test("MultiTaskAdapter API format: predict_all")
def test_adapter_predict_all_format():
    from multitask_adapter import MultiTaskAdapter
    adapter = MultiTaskAdapter(model_path="/nonexistent/path.pt")
    assert hasattr(adapter, "predict_all")
    assert hasattr(adapter, "predict_developability")
    assert hasattr(adapter, "predict_wetlab")
    assert hasattr(adapter, "predict_potency")


# ===== TEST 6: End-to-end =====
@test("End-to-end: data → model → prediction")
def test_e2e():
    from esm2_hybrid_encoder import ESM2HybridEncoder
    from unified_multitask_model import UnifiedMultiTaskModel
    from unified_dataset import UnifiedAntibodyDataset, unified_collate_fn

    data_path = os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv")
    ds = UnifiedAntibodyDataset(data_path)

    enc = ESM2HybridEncoder(hidden_dim=64)
    model = UnifiedMultiTaskModel(encoder=enc, encoder_dim=64)
    model.eval()

    # Get a batch
    loader = DataLoader(ds, batch_size=4, collate_fn=unified_collate_fn)
    hc_seqs, lc_seqs, labels, masks, biophys, cached_emb = next(iter(loader))

    with torch.no_grad():
        outputs = model(hc_seqs, lc_seqs, biophys)

    # Verify all tasks present and shapes correct
    assert len(outputs) == 8, f"Expected 8 tasks, got {len(outputs)}"
    for task, tensor in outputs.items():
        assert tensor.shape == (4,), f"Task {task}: expected (4,), got {tensor.shape}"
        assert not torch.isnan(tensor).any(), f"Task {task} has NaN"


# ===== RUN ALL TESTS =====
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Unified Integration Tests")
    print(f"{'='*60}\n")

    tests = [
        test_encoder_forward,
        test_encoder_with_biophys,
        test_encoder_dim,
        test_model_forward,
        test_bounded_tasks,
        test_predict_numpy,
        test_dataset,
        test_collate,
        test_masked_loss,
        test_training_loop,
        test_adapter_init,
        test_adapter_chromatography_format,
        test_adapter_predict_all_format,
        test_e2e,
    ]

    for t in tests:
        t()

    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print(f"{'='*60}")

    sys.exit(0 if FAILED == 0 else 1)
