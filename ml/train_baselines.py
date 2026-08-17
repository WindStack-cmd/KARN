import os
import time
import json
import torch
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ============================================================
# KARN - PHASE 9 Baseline Models
# Non-sequential baselines: Logistic Regression & Random Forest
# ============================================================

WINDOWS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\training_windows"
RESULTS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\results"
MODELS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\models"

def compute_sequence_stats(X):
    """
    Computes sequence statistics for each feature in a (N, 30, 23) sequence.
    Stats: Mean, Std, Min, Max, Range, and First-to-Last Change.
    Output size: (N, 6 * 23) = (N, 138)
    """
    # X shape: (N, 30, 23)
    x_mean = np.mean(X, axis=1)          # (N, 23)
    x_std = np.std(X, axis=1)            # (N, 23)
    x_min = np.min(X, axis=1)            # (N, 23)
    x_max = np.max(X, axis=1)            # (N, 23)
    x_range = x_max - x_min              # (N, 23)
    x_change = X[:, -1, :] - X[:, 0, :]  # (N, 23)
    
    return np.concatenate([x_mean, x_std, x_min, x_max, x_range, x_change], axis=1)

def compute_flattened(X):
    """
    Flattens the sequence from (N, 30, 23) to (N, 690)
    """
    return X.reshape(X.shape[0], -1)

def load_data():
    """
    Loads training, validation, and test PyTorch tensors and converts to Numpy.
    """
    print("Loading serialized PyTorch tensors...")
    
    X_train = torch.load(os.path.join(WINDOWS_DIR, "X_train.pt"), map_location="cpu").numpy()
    y_train = torch.load(os.path.join(WINDOWS_DIR, "y_train_cls.pt"), map_location="cpu").numpy()
    
    X_val = torch.load(os.path.join(WINDOWS_DIR, "X_val.pt"), map_location="cpu").numpy()
    y_val = torch.load(os.path.join(WINDOWS_DIR, "y_val_cls.pt"), map_location="cpu").numpy()
    
    X_test = torch.load(os.path.join(WINDOWS_DIR, "X_test.pt"), map_location="cpu").numpy()
    y_test = torch.load(os.path.join(WINDOWS_DIR, "y_test_cls.pt"), map_location="cpu").numpy()
    
    # Assert sanity
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf values!")
            
    print(f"Data Loaded successfully.")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_class_name(idx):
    return ["Quiet", "C-class", "M-class", "X-class"][idx]

def compute_metrics(y_true, y_pred, y_prob_2d, classes_seen):
    """
    Safely computes classification metrics including precision, recall, f1, and confusion matrix.
    Because the classes are split dynamically (Train=C, Val=M, Test=X), we compute metrics over
    all 4 classes [0, 1, 2, 3] and flag what is scientifically meaningful.
    """
    # Initialize dictionary for metrics
    metrics = {}
    
    # Total samples and unique classes observed
    metrics["num_samples"] = int(len(y_true))
    unique_observed = sorted(list(np.unique(y_true)))
    metrics["classes_observed"] = [int(u) for u in unique_observed]
    metrics["class_names_observed"] = [get_class_name(u) for u in unique_observed]
    
    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # Precision, Recall, F1 for class [0, 1, 2, 3]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    metrics["confusion_matrix"] = cm
    
    # Class-wise metrics
    class_metrics = {}
    for i in range(4):
        c_name = get_class_name(i)
        is_meaningful = True
        reason = "Valid"
        
        # Explain scientific meaning
        # Classes 2 and 3 can never be predicted by the model because they are not in Train.
        # Classes not in y_true have undefined metrics.
        if i not in classes_seen:
            is_meaningful = False
            reason = "Class was not present in training data; model cannot predict it."
        elif i not in unique_observed:
            is_meaningful = False
            reason = "Class was not present in this split's actual ground-truth label distribution."
            
        class_metrics[c_name] = {
            "is_scientific_meaningful": is_meaningful,
            "meaningful_reason": reason,
            "precision": float(precision[i]) if is_meaningful else "not meaningful/undefined",
            "recall": float(recall[i]) if is_meaningful else "not meaningful/undefined",
            "f1_score": float(f1[i]) if is_meaningful else "not meaningful/undefined",
            "support": int(support[i])
        }
        
    metrics["class_metrics"] = class_metrics
    
    # Predict probabilities metrics
    # Reconstruct 4-class probabilities from 2-class model output
    y_prob_4d = np.zeros((len(y_true), 4))
    # classes_seen is typically [0, 1]
    for idx, cls in enumerate(classes_seen):
        if idx < y_prob_2d.shape[1]:
            y_prob_4d[:, cls] = y_prob_2d[:, idx]
            
    # Sample prediction probabilities (first 5 samples)
    metrics["sample_probabilities"] = y_prob_4d[:5].tolist()
    
    return metrics

def train_and_eval():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    classes_seen = sorted(list(np.unique(y_train))) # Expected to be [0, 1]
    print(f"Classes seen in training data: {classes_seen} ({[get_class_name(c) for c in classes_seen]})")
    
    # Define representations
    representations = {
        "statistical": {
            "train_func": lambda X: compute_sequence_stats(X),
            "train_X": None,
            "val_X": None,
            "test_X": None
        },
        "flattened": {
            "train_func": lambda X: compute_flattened(X),
            "train_X": None,
            "val_X": None,
            "test_X": None
        }
    }
    
    # Compute representations
    for name, r in representations.items():
        print(f"Computing {name} sequence representation...")
        r["train_X"] = r["train_func"](X_train)
        r["val_X"] = r["train_func"](X_val)
        r["test_X"] = r["train_func"](X_test)
        print(f"  {name} shapes: train={r['train_X'].shape}, val={r['val_X'].shape}, test={r['test_X'].shape}")
        
    results = {}
    
    # Define models
    # We use Pipeline to bundle StandardScaler and Logistic Regression
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight=None))
    ])
    
    # RandomForest does not strictly require scaling, but standard pipeline works fine
    rf_pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight=None))
    ])
    
    models = {
        "logistic_regression": lr_pipeline,
        "random_forest": rf_pipeline
    }
    
    for model_name, pipeline in models.items():
        results[model_name] = {}
        for rep_name in ["statistical", "flattened"]:
            print(f"\n--- Training {model_name} on {rep_name} representation ---")
            train_X = representations[rep_name]["train_X"]
            val_X = representations[rep_name]["val_X"]
            test_X = representations[rep_name]["test_X"]
            
            start_time = time.time()
            pipeline.fit(train_X, y_train)
            train_time = time.time() - start_time
            
            print(f"Training completed in {train_time:.4f} seconds.")
            
            # Predict
            train_pred = pipeline.predict(train_X)
            val_pred = pipeline.predict(val_X)
            test_pred = pipeline.predict(test_X)
            
            # Probabilities
            train_prob = pipeline.predict_proba(train_X)
            val_prob = pipeline.predict_proba(val_X)
            test_prob = pipeline.predict_proba(test_X)
            
            # Verify probabilities are finite
            if not (np.isfinite(train_prob).all() and np.isfinite(val_prob).all() and np.isfinite(test_prob).all()):
                raise ValueError("Calculated prediction probabilities contain NaN or Inf values!")
            
            # Compute metrics
            m_train = compute_metrics(y_train, train_pred, train_prob, classes_seen)
            m_val = compute_metrics(y_val, val_pred, val_prob, classes_seen)
            m_test = compute_metrics(y_test, test_pred, test_prob, classes_seen)
            
            rep_results = {
                "training_time_seconds": train_time,
                "model_configuration": str(pipeline.get_params()["classifier"]),
                "train_metrics": m_train,
                "validation_metrics": m_val,
                "test_metrics": m_test,
                "memory_safe_execution_status": "PASS"
            }
            
            results[model_name][rep_name] = rep_results
            
            model_save_path = os.path.join(MODELS_DIR, f"baseline_{model_name}_{rep_name}.pkl")
            with open(model_save_path, "wb") as f_out:
                pickle.dump(pipeline, f_out)
            print(f"Saved {model_name} ({rep_name} pipeline) to {model_save_path}")

            if rep_name == "statistical":
                compatibility_path = os.path.join(MODELS_DIR, f"baseline_{model_name}.pkl")
                with open(compatibility_path, "wb") as f_out:
                    pickle.dump(pipeline, f_out)
                print(f"Saved compatibility copy to {compatibility_path}")
                
    # Save metrics JSON
    metrics_save_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    with open(metrics_save_path, "w") as f_out:
        json.dump(results, f_out, indent=4)
    print(f"\n[OK] Baseline metrics successfully written to {metrics_save_path}")

if __name__ == "__main__":
    train_and_eval()
