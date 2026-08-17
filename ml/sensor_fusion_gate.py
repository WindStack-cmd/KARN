import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# KARN - PHASE 11 Context-Aware Dual-Sensor Fusion Gate
# ============================================================

FEATURES_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"

def detect_sensor_features():
    """
    Loads features dataset and groups columns by sensor family dynamically.
    Returns:
      solexs_cols: list of names
      hel1os_cols: list of names
      context_cols: list of names of other features
    """
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found at: {FEATURES_PATH}")
        
    df = pd.read_csv(FEATURES_PATH)
    
    # Identify feature columns (excluding label indicators and raw times)
    label_cols = ["flare_label", "flare_class", "time_to_peak", "event_id"]
    excluded_cols = ["time_seconds", "timestamp"]
    
    feature_cols = [c for c in df.columns if c not in label_cols and c not in excluded_cols]
    
    solexs_cols = [c for c in feature_cols if "solexs" in c.lower()]
    hel1os_cols = [c for c in feature_cols if "hel1os" in c.lower()]
    context_cols = [c for c in feature_cols if c not in solexs_cols and c not in hel1os_cols]
    
    print("\n--- Feature Detection Verification ---")
    print(f"Total model features detected: {len(feature_cols)}")
    print(f"SoLEXS features ({len(solexs_cols)}): {solexs_cols}")
    print(f"HEL1OS features ({len(hel1os_cols)}): {hel1os_cols}")
    print(f"Context / Other features ({len(context_cols)}): {context_cols}")
    print("--------------------------------------\n")
    
    # Safety checks
    assert len(solexs_cols) > 0, "No SoLEXS features detected!"
    assert len(hel1os_cols) > 0, "No HEL1OS features detected!"
    assert len(feature_cols) == len(solexs_cols) + len(hel1os_cols) + len(context_cols), "Features count mismatch!"
    
    return feature_cols, solexs_cols, hel1os_cols, context_cols


class ContextAwareFusionGate(nn.Module):
    def __init__(self, solexs_dim=10, hel1os_dim=10, context_dim=3, projection_dim=16):
        """
        Lightweight Differentiable Context-Aware Weighted Gate.
        Projects both sensor representations to projection_dim, 
        and decides fusion weights dynamically based on overall context and inputs.
        """
        super().__init__()
        self.solexs_dim = solexs_dim
        self.hel1os_dim = hel1os_dim
        self.context_dim = context_dim
        self.projection_dim = projection_dim
        
        # Sensor projections
        self.proj_solexs = nn.Linear(solexs_dim, projection_dim)
        self.proj_hel1os = nn.Linear(hel1os_dim, projection_dim)
        
        # Gating network
        # Total gate input dim: solexs_dim + hel1os_dim + context_dim
        gate_input_dim = solexs_dim + hel1os_dim + context_dim
        
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # Raw logits for [SoLEXS, HEL1OS] weights
        )
        
    def forward(self, x_solexs, x_hel1os, x_context):
        """
        Differentiable forward pass.
        Args:
          x_solexs: torch.Tensor of shape (B, L, solexs_dim) or (B, solexs_dim)
          x_hel1os: torch.Tensor of shape (B, L, hel1os_dim) or (B, hel1os_dim)
          x_context: torch.Tensor of shape (B, L, context_dim) or (B, context_dim)
        Returns:
          fused_representation: shape (..., projection_dim)
          w_solexs: shape (..., 1) -> normalized weight for SoLEXS
          w_hel1os: shape (..., 1) -> normalized weight for HEL1OS
        """
        if x_solexs.shape[:-1] != x_hel1os.shape[:-1] or x_solexs.shape[:-1] != x_context.shape[:-1]:
            raise ValueError(
                "Input leading dimensions must match across SoLEXS, HEL1OS, and context tensors."
            )
        if x_solexs.shape[-1] != self.solexs_dim:
            raise ValueError(
                f"SoLEXS feature dimension mismatch: expected {self.solexs_dim}, got {x_solexs.shape[-1]}"
            )
        if x_hel1os.shape[-1] != self.hel1os_dim:
            raise ValueError(
                f"HEL1OS feature dimension mismatch: expected {self.hel1os_dim}, got {x_hel1os.shape[-1]}"
            )
        if x_context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Context feature dimension mismatch: expected {self.context_dim}, got {x_context.shape[-1]}"
            )

        # 1. Project both sensor features
        h_solexs = self.proj_solexs(x_solexs)  # Shape: (..., projection_dim)
        h_hel1os = self.proj_hel1os(x_hel1os)  # Shape: (..., projection_dim)
        
        # 2. Concat all inputs for calculating context-aware weights
        gate_in = torch.cat([x_solexs, x_hel1os, x_context], dim=-1)
        
        # 3. Compute logits and apply Softmax for numerical stability
        logits = self.gate_net(gate_in)          # Shape: (..., 2)
        weights = F.softmax(logits, dim=-1)     # Shape: (..., 2)
        
        w_solexs = weights[..., 0:1]           # Shape: (..., 1)
        w_hel1os = weights[..., 1:2]           # Shape: (..., 1)
        
        # 4. Compute fused representation
        fused = w_solexs * h_solexs + w_hel1os * h_hel1os
        
        return fused, w_solexs, w_hel1os


def run_sanity_tests():
    """
    Runs simulated sensor disturbances to verify gate outputs and weights.
    """
    print("=== STARTING FUSION GATE SANITY TESTS ===")
    
    # 1. Feature detection
    feature_cols, solexs_cols, hel1os_cols, context_cols = detect_sensor_features()
    
    # 2. Gate dimensions
    solexs_dim = len(solexs_cols)
    hel1os_dim = len(hel1os_cols)
    context_dim = len(context_cols)
    projection_dim = 16
    
    # Initialize the gate with fixed seed for determinism in testing
    torch.manual_seed(42)
    gate = ContextAwareFusionGate(
        solexs_dim=solexs_dim, 
        hel1os_dim=hel1os_dim, 
        context_dim=context_dim, 
        projection_dim=projection_dim
    )
    gate.eval()
    
    B, L = 4, 30  # Batch size 4, lookback 30 minutes
    
    cases = {
        "Case 1: Normal Background": {
            "solexs_val": 0.1,
            "hel1os_val": 0.1,
            "context_val": 0.0
        },
        "Case 2: HEL1OS-dominant Disturbance": {
            "solexs_val": 0.1,
            "hel1os_val": 5.0,  # High flux in HEL1OS
            "context_val": 1.0  # active flags
        },
        "Case 3: SoLEXS-dominant Change": {
            "solexs_val": 5.0,  # High flux in SoLEXS
            "hel1os_val": 0.1,
            "context_val": 1.0  # active flags
        },
        "Case 4: Correlated Change (Both Sensors)": {
            "solexs_val": 4.5,
            "hel1os_val": 4.5,
            "context_val": 1.0
        }
    }
    
    case_results = {}
    
    for case_name, vals in cases.items():
        print(f"\n--- Running {case_name} ---")
        
        # Construct tensors
        x_solexs = torch.full((B, L, solexs_dim), vals["solexs_val"], dtype=torch.float32)
        x_hel1os = torch.full((B, L, hel1os_dim), vals["hel1os_val"], dtype=torch.float32)
        x_context = torch.full((B, L, context_dim), vals["context_val"], dtype=torch.float32)
        
        # Forward pass
        with torch.no_grad():
            fused, w_solexs, w_hel1os = gate(x_solexs, x_hel1os, x_context)
            
        # Verify shapes
        assert fused.shape == (B, L, projection_dim), f"Fused representation shape mismatch! Expected: {(B, L, projection_dim)}, got: {fused.shape}"
        assert w_solexs.shape == (B, L, 1), f"SoLEXS weight shape mismatch! Expected: {(B, L, 1)}, got: {w_solexs.shape}"
        assert w_hel1os.shape == (B, L, 1), f"HEL1OS weight shape mismatch! Expected: {(B, L, 1)}, got: {w_hel1os.shape}"
        
        # Verify no NaN or Inf
        assert torch.isfinite(fused).all(), f"Fused representation contains NaN or Inf!"
        assert torch.isfinite(w_solexs).all(), f"SoLEXS weight contains NaN or Inf!"
        assert torch.isfinite(w_hel1os).all(), f"HEL1OS weight contains NaN or Inf!"
        
        # Verify weights bounds and sum-to-1 constraint
        assert (w_solexs >= 0.0).all() and (w_solexs <= 1.0).all(), "SoLEXS weight out of bounds!"
        assert (w_hel1os >= 0.0).all() and (w_hel1os <= 1.0).all(), "HEL1OS weight out of bounds!"
        
        sums = w_solexs + w_hel1os
        assert torch.allclose(sums, torch.ones_like(sums)), "Weights do not sum to 1.0!"
        
        # Compute mean weights for print
        mean_w_solexs = w_solexs.mean().item()
        mean_w_hel1os = w_hel1os.mean().item()
        
        print(f"  Fused output shape      : {list(fused.shape)}")
        print(f"  Weights output shape    : SoLEXS={list(w_solexs.shape)}, HEL1OS={list(w_hel1os.shape)}")
        print(f"  Mean Gating Weights     : SoLEXS = {mean_w_solexs:.4f}, HEL1OS = {mean_w_hel1os:.4f}")
        print(f"  Weights Sum Validation  : {sums.mean().item():.2f}")
        print("  Status                  : PASS")
        
        case_results[case_name] = {
            "mean_w_solexs": mean_w_solexs,
            "mean_w_hel1os": mean_w_hel1os,
            "status": "PASS"
        }
        
    print("\n===========================================")
    print("FUSION GATE SANITY CHECKS: ALL PASSED")
    print("===========================================")
    return case_results

if __name__ == "__main__":
    run_sanity_tests()
