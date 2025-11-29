# ml_service/api.py

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import asyncio
import threading
import time

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from inference import FraudModelService
from config import DATA_DIR, CSV_PATH


# ------------------------
# Paths
# ------------------------

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_LOG_PATH = ARTIFACTS_DIR / "feedback_log.jsonl"  # line-delimited JSON


# ------------------------
# Pydantic models
# ------------------------

class PredictRequest(BaseModel):
    """
    Request payload for /predict.

    The 'features' dict must contain ALL the numeric features
    expected by the model, with names matching those in feature_names.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Mapping from feature name to numeric value.",
    )


class PredictResponse(BaseModel):
    prob_not_fraud: float = Field(
        ...,
        description="P(y=0 | x): probability that the transaction is NOT fraud.",
    )
    prob_fraud: float = Field(
        ...,
        description="P(y=1 | x): probability that the transaction IS fraud.",
    )
    is_fraud: int = Field(
        ...,
        description="Binary prediction (1=fraud, 0=not fraud) after applying threshold.",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used by the model on P(y=1 | x).",
    )


class FeedbackRequest(BaseModel):
    """
    Feedback payload for online learning.

    - features: same dict you sent to /predict
    - model_prob_fraud: P(y=1 | x) returned by /predict
    - model_prob_not_fraud: P(y=0 | x) returned by /predict
    - model_label: label returned by /predict (0/1)
    - true_label: actual ground truth (0=not fraud, 1=fraud)
    - metadata: optional extra info (user id, transaction id, etc.)
    """
    features: Dict[str, float]
    model_prob_fraud: float
    model_prob_not_fraud: float
    model_label: int
    true_label: int
    metadata: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


# ------------------------
# FastAPI app + model service
# ------------------------

app = FastAPI(
    title="Fraud Detection Service",
    description="ML backend for credit card fraud detection (Phase V).",
    version="1.0.0",
)

# Load the trained model once at startup
service = FraudModelService()

# In-memory cache to store predictions for feedback lookup
# Key: transaction_id, Value: {features, model_prob_fraud, model_prob_not_fraud, model_label}
_prediction_cache: Dict[str, Dict[str, Any]] = {}

# Online learning state tracking
_online_learning_state = {
    "last_feedback_count": 0,  # Number of feedback samples processed last time
    "last_update_time": datetime.now(),  # When we last ran online SGD
    "is_updating": False,  # Flag to prevent concurrent updates
}

# ------------------------
# Preprocessing utilities
# ------------------------

def load_preprocessing_stats():
    """Load statistics needed for preprocessing from the processed dataset."""
    try:
        # Load original data to get Amount_float statistics for z-score calculation
        df_original = pd.read_csv(DATA_DIR / "final_dataset.csv")
        
        # Load processed data to get MCC codes list
        df_processed = pd.read_csv(CSV_PATH)
        
        # Load normalization stats for avg_transaction_per_user and avg_transaction_per_card
        normalization_stats_path = ARTIFACTS_DIR / "normalization_stats.json"
        norm_stats = {}
        if normalization_stats_path.exists():
            with open(normalization_stats_path, "r") as f:
                norm_stats = json.load(f)
        else:
            print(f"[WARNING] Normalization stats not found at {normalization_stats_path}")
        
        stats = {
            # Amount_float statistics for z-score normalization
            # Calculate z-score as (Amount_float - mean) / std
            "amount_float_mean": df_original["Amount_float"].mean(),
            "amount_float_std": df_original["Amount_float"].std(),
            
            # seconds_1990 scaler parameters (MinMaxScaler)
            "seconds_1990_min": df_original["seconds_1990"].min(),
            "seconds_1990_max": df_original["seconds_1990"].max(),
            
            # Get all MCC codes from feature names
            "mcc_codes": [col for col in df_processed.columns if col.startswith("MCC_")],
            
            # Average transaction normalization stats (MinMax)
            "avg_transaction_per_user_min": norm_stats.get("avg_transaction_per_user", {}).get("min", 0.0),
            "avg_transaction_per_user_max": norm_stats.get("avg_transaction_per_user", {}).get("max", 1.0),
            "avg_transaction_per_card_min": norm_stats.get("avg_transaction_per_card", {}).get("min", 0.0),
            "avg_transaction_per_card_max": norm_stats.get("avg_transaction_per_card", {}).get("max", 1.0),
        }
        
        print(f"[PREPROC] Loaded preprocessing stats:")
        print(f"  - Amount_float: mean={stats['amount_float_mean']:.2f}, std={stats['amount_float_std']:.2f}")
        print(f"  - seconds_1990: min={stats['seconds_1990_min']:.0f}, max={stats['seconds_1990_max']:.0f}")
        print(f"  - avg_transaction_per_user: min={stats['avg_transaction_per_user_min']:.6f}, max={stats['avg_transaction_per_user_max']:.6f}")
        print(f"  - avg_transaction_per_card: min={stats['avg_transaction_per_card_min']:.6f}, max={stats['avg_transaction_per_card_max']:.6f}")
        print(f"  - MCC codes: {len(stats['mcc_codes'])} codes")
        
        return stats
    except Exception as e:
        print(f"[WARNING] Could not load preprocessing stats: {e}")
        print("[WARNING] Using default values - predictions may be inaccurate!")
        # Return defaults - these should be updated with actual values
        return {
            "amount_float_mean": 0.0,
            "amount_float_std": 1.0,
            "seconds_1990_min": 0.0,
            "seconds_1990_max": 1.0,
            "mcc_codes": [],
            "avg_transaction_per_user_min": 0.0,
            "avg_transaction_per_user_max": 1.0,
            "avg_transaction_per_card_min": 0.0,
            "avg_transaction_per_card_max": 1.0,
        }

# Load preprocessing statistics at startup
_preprocessing_stats = load_preprocessing_stats()

def calculate_seconds_1990(dt: datetime) -> float:
    """Calculate seconds since 1990-01-01 00:00:00 UTC."""
    epoch_1990 = datetime(1990, 1, 1, 0, 0, 0)
    delta = dt - epoch_1990
    return delta.total_seconds()

def normalize_seconds_1990(seconds: float) -> float:
    """Normalize seconds_1990 using MinMaxScaler parameters."""
    min_val = _preprocessing_stats["seconds_1990_min"]
    max_val = _preprocessing_stats["seconds_1990_max"]
    if max_val == min_val:
        return 0.0
    return (seconds - min_val) / (max_val - min_val)

def calculate_amount_zscore(amount: float) -> float:
    """Calculate Amount_zscore from Amount_float using z-score normalization."""
    mean = _preprocessing_stats["amount_float_mean"]
    std = _preprocessing_stats["amount_float_std"]
    if std == 0:
        return 0.0
    return (amount - mean) / std

def normalize_avg_transaction_per_user(raw_value: float) -> float:
    """Normalize average transaction per user using MinMax normalization."""
    min_val = _preprocessing_stats["avg_transaction_per_user_min"]
    max_val = _preprocessing_stats["avg_transaction_per_user_max"]
    if max_val == min_val:
        return 0.0
    return (raw_value - min_val) / (max_val - min_val)

def normalize_avg_transaction_per_card(raw_value: float) -> float:
    """Normalize average transaction per card using MinMax normalization."""
    min_val = _preprocessing_stats["avg_transaction_per_card_min"]
    max_val = _preprocessing_stats["avg_transaction_per_card_max"]
    if max_val == min_val:
        return 0.0
    return (raw_value - min_val) / (max_val - min_val)

def one_hot_encode_mcc(mcc: int) -> Dict[str, float]:
    """One-hot encode MCC code."""
    mcc_codes = _preprocessing_stats["mcc_codes"]
    result = {code: 0.0 for code in mcc_codes}
    mcc_key = f"MCC_{mcc}"
    if mcc_key in result:
        result[mcc_key] = 1.0
    return result

def one_hot_encode_country(country_development: Union[str, int]) -> Dict[str, float]:
    """One-hot encode country development category.
    
    Accepts either:
    - String: 'developed', 'developing', 'underdeveloped'
    - Integer: 2 (developed), 1 (developing), 0 (underdeveloped)
    """
    # Handle integer encoding (from client.py mapping)
    if isinstance(country_development, int):
        int_mapping = {
            2: {'DevCat_Developed': 1.0, 'DevCat_Developing': 0.0, 'DevCat_UnderDeveloped': 0.0},
            1: {'DevCat_Developed': 0.0, 'DevCat_Developing': 1.0, 'DevCat_UnderDeveloped': 0.0},
            0: {'DevCat_Developed': 0.0, 'DevCat_Developing': 0.0, 'DevCat_UnderDeveloped': 1.0},
        }
        return int_mapping.get(country_development, {'DevCat_Developed': 0.0, 'DevCat_Developing': 1.0, 'DevCat_UnderDeveloped': 0.0})
    
    # Handle string encoding
    if isinstance(country_development, str):
        str_mapping = {
            'developed': {'DevCat_Developed': 1.0, 'DevCat_Developing': 0.0, 'DevCat_UnderDeveloped': 0.0},
            'developing': {'DevCat_Developed': 0.0, 'DevCat_Developing': 1.0, 'DevCat_UnderDeveloped': 0.0},
            'underdeveloped': {'DevCat_Developed': 0.0, 'DevCat_Developing': 0.0, 'DevCat_UnderDeveloped': 1.0},
        }
        return str_mapping.get(country_development.lower(), {'DevCat_Developed': 0.0, 'DevCat_Developing': 1.0, 'DevCat_UnderDeveloped': 0.0})
    
    # Default to developing if unknown type
    return {'DevCat_Developed': 0.0, 'DevCat_Developing': 1.0, 'DevCat_UnderDeveloped': 0.0}

def preprocess_client_input(client_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Transform client input format to model's expected feature format.
    
    Client sends:
    - amount_float, amount_is_refund
    - avg_transaction_per_card (raw, unnormalized), avg_transaction_per_user (raw, unnormalized)
    - country_development (string)
    - transaction_datetime (ISO string)
    - mcc (integer)
    - has_error, online_flag
    
    Model expects:
    - Amount_zscore (calculated from amount_float)
    - Amount_IsRefund
    - avg_transaction_per_card_norm, avg_transaction_per_user_norm
    - seconds_1990 (calculated and normalized from transaction_datetime)
    - DevCat_* (one-hot encoded from country_development)
    - MCC_* (one-hot encoded from mcc)
    - OnlineFlag, HasError
    - ErrBin_* features
    - UseChip_* features
    """
    # Start with direct mappings
    features = {
        "Amount_IsRefund": float(client_data.get("Amount_IsRefund", client_data.get("amount_is_refund", 0))),
        "OnlineFlag": float(client_data.get("OnlineFlag", client_data.get("online_flag", 0))),
        "HasError": float(client_data.get("HasError", client_data.get("has_error", 0))),
    }
    
    # Normalize average transaction per user and per card from raw values
    # User provides raw (unnormalized) values, we normalize them here
    avg_transaction_per_user_raw = client_data.get("avg_transaction_per_user", 
                                                   client_data.get("avg_transaction_per_user_norm", 0.0))
    avg_transaction_per_card_raw = client_data.get("avg_transaction_per_card",
                                                   client_data.get("avg_transaction_per_card_norm", 0.0))
    
    features["avg_transaction_per_user_norm"] = normalize_avg_transaction_per_user(float(avg_transaction_per_user_raw))
    features["avg_transaction_per_card_norm"] = normalize_avg_transaction_per_card(float(avg_transaction_per_card_raw))
    
    # Calculate Amount_zscore from Amount_float
    amount_float = client_data.get("Amount_float", client_data.get("amount_float", 0.0))
    features["Amount_zscore"] = calculate_amount_zscore(float(amount_float))
    
    # Calculate and normalize seconds_1990 from transaction_datetime
    transaction_datetime = client_data.get("transaction_datetime")
    if transaction_datetime:
        try:
            # Parse datetime
            dt_str = str(transaction_datetime).replace('Z', '+00:00')
            dt = datetime.fromisoformat(dt_str)
            seconds = calculate_seconds_1990(dt)
            features["seconds_1990"] = normalize_seconds_1990(seconds)
        except Exception as e:
            # Default to current time if parsing fails
            dt = datetime.now()
            seconds = calculate_seconds_1990(dt)
            features["seconds_1990"] = normalize_seconds_1990(seconds)
    else:
        # Default to current time
        dt = datetime.now()
        seconds = calculate_seconds_1990(dt)
        features["seconds_1990"] = normalize_seconds_1990(seconds)
    
    # One-hot encode country development
    # Client sends it as integer (0, 1, 2) or string ("developed", "developing", "underdeveloped")
    country_dev = client_data.get("country_development", 1)  # Default to 1 (developing)
    country_features = one_hot_encode_country(country_dev)
    features.update(country_features)
    
    # One-hot encode MCC
    mcc = client_data.get("MCC", client_data.get("mcc", 0))
    mcc_features = one_hot_encode_mcc(int(mcc))
    features.update(mcc_features)
    
    # Ensure all required features from model are present (set missing ones to 0)
    for feature_name in service.feature_names:
        if feature_name not in features:
            features[feature_name] = 0.0
    
    return features


# ------------------------
# Endpoints
# ------------------------

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "detail": "Fraud model service is running."}


@app.post("/predict")
def predict(req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Predict fraud probability and label for a single transaction.

    Accepts either:
    - Flat dict from client.py: {"amount_float": 100.0, "mcc": 5411, "transaction_datetime": "...", ...}
    - Wrapped dict: {"features": {"Amount_zscore": 1.2, "seconds_1990": 0.5, ...}}
    """
    try:
        # Handle both formats
        if "features" in req and isinstance(req["features"], dict):
            # Already in model format (wrapped dict)
            features = req["features"]
        else:
            # Client format - needs preprocessing
            features = preprocess_client_input(req)
        
        result = service.predict_single(features)
        
        # Generate transaction ID
        import uuid
        transaction_id = str(uuid.uuid4())
        
        # Store prediction data for feedback lookup
        _prediction_cache[transaction_id] = {
            "features": features.copy(),  # Store preprocessed features
            "model_prob_fraud": result["prob_fraud"],
            "model_prob_not_fraud": result["prob_not_fraud"],
            "model_label": result["is_fraud"],
        }
        
        # Return format compatible with client.py expectations
        return {
            "prediction": result["is_fraud"],
            "confidence": result["prob_fraud"],
            "transaction_id": transaction_id,
            # Also include full response for API docs compatibility
            "prob_not_fraud": result["prob_not_fraud"],
            "prob_fraud": result["prob_fraud"],
            "is_fraud": result["is_fraud"],
            "threshold": result["threshold"],
        }
    except KeyError as e:
        # Missing or wrong feature name
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Any other unexpected error
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/feedback")
def submit_feedback(req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Receive user feedback / ground truth for a past prediction.
    
    Saves feedback in the format required by online_sgd_update.py:
    - features: dict with all feature values
    - true_label: 0 or 1 (actual ground truth)
    - model_prob_fraud, model_prob_not_fraud, model_label: for reference
    - metadata: optional additional info

    Accepts two formats:
    1. Simple format (from client.py): {"transaction_id": "...", "is_correct": true/false}
       - Looks up stored prediction data and converts to detailed format
    2. Detailed format: Full FeedbackRequest with features, probabilities, true_label, etc.
    """
    record = None
    
    # Handle simple format from client.py
    if "transaction_id" in req and "is_correct" in req:
        transaction_id = req["transaction_id"]
        is_correct = bool(req["is_correct"])
        
        # Look up stored prediction data
        if transaction_id not in _prediction_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction ID '{transaction_id}' not found. Prediction data may have expired.",
            )
        
        cached_data = _prediction_cache[transaction_id]
        model_label = cached_data["model_label"]
        
        # Calculate true_label from is_correct
        # If prediction was correct, true_label = model_label
        # If prediction was incorrect, true_label = 1 - model_label
        if is_correct:
            true_label = model_label
        else:
            true_label = 1 - model_label
        
        # Create detailed format record matching API documentation
        record = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "features": cached_data["features"],
            "model_prob_fraud": cached_data["model_prob_fraud"],
            "model_prob_not_fraud": cached_data["model_prob_not_fraud"],
            "model_label": model_label,
            "true_label": int(true_label),
            "metadata": {
                "transaction_id": transaction_id,
                "is_correct": is_correct,
            },
        }
        
        # Clean up cache (optional - could keep for a while)
        # del _prediction_cache[transaction_id]
        
    # Handle detailed format (matches API documentation)
    elif "features" in req and "true_label" in req:
        record = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "features": req.get("features", {}),
            "model_prob_fraud": float(req.get("model_prob_fraud", 0.0)),
            "model_prob_not_fraud": float(req.get("model_prob_not_fraud", 0.0)),
            "model_label": int(req.get("model_label", 0)),
            "true_label": int(req.get("true_label", 0)),
            "metadata": req.get("metadata", {}),
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid feedback format. Expected either {'transaction_id', 'is_correct'} or detailed format with 'features' and 'true_label'.",
        )

    # Validate that record has required fields for online learning
    if "features" not in record or "true_label" not in record:
        raise HTTPException(
            status_code=500,
            detail="Internal error: Failed to create valid feedback record.",
        )

    try:
        # Append to JSONL file (format expected by online_sgd_update.py)
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write feedback log: {e}",
        )

    return {
        "status": "ok",
        "message": "Feedback recorded successfully.",
    }


# ------------------------
# Online Learning Background Task
# ------------------------

def count_feedback_samples() -> int:
    """Count the number of feedback samples in the log file."""
    if not FEEDBACK_LOG_PATH.exists():
        return 0
    
    count = 0
    try:
        with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # Only count records with features and true_label (usable for training)
                    if "features" in rec and "true_label" in rec:
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ONLINE-LEARNING] Error counting feedback: {e}")
    
    return count

def run_online_sgd_update():
    """Run online SGD update by calling the online_sgd_update.py main function."""
    if _online_learning_state["is_updating"]:
        print("[ONLINE-LEARNING] Update already in progress, skipping...")
        return
    
    _online_learning_state["is_updating"] = True
    try:
        print("[ONLINE-LEARNING] Starting online SGD update...")
        
        # Import and run the online SGD update
        from online_sgd_update import main as online_sgd_main
        
        # Run in a separate thread to avoid blocking
        online_sgd_main()
        
        # Reload the model after update
        print("[ONLINE-LEARNING] Reloading model after update...")
        global service
        service = FraudModelService()
        
        # Update state
        current_count = count_feedback_samples()
        _online_learning_state["last_feedback_count"] = current_count
        _online_learning_state["last_update_time"] = datetime.now()
        
        print(f"[ONLINE-LEARNING] Online SGD update completed. Processed {current_count} feedback samples.")
        
    except Exception as e:
        print(f"[ONLINE-LEARNING] Error during online SGD update: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _online_learning_state["is_updating"] = False

def check_and_run_online_learning():
    """Check if conditions are met to run online learning, then run it."""
    current_count = count_feedback_samples()
    last_count = _online_learning_state["last_feedback_count"]
    last_update = _online_learning_state["last_update_time"]
    now = datetime.now()
    
    # Check conditions:
    # 1. At least 1 feedback sample exists
    # 2. Either: 15 minutes have passed OR 67 new samples since last update
    time_since_update = (now - last_update).total_seconds() / 60  # minutes
    new_samples = current_count - last_count
    
    should_run = False
    reason = ""
    
    if current_count == 0:
        # No feedback samples yet
        return
    
    if time_since_update >= 15:
        should_run = True
        reason = f"15 minutes have passed (last update: {time_since_update:.1f} min ago)"
    elif new_samples >= 67:
        should_run = True
        reason = f"67 new feedback samples received ({new_samples} new samples)"
    
    if should_run:
        print(f"[ONLINE-LEARNING] Triggering update: {reason}")
        # Run in a separate thread to avoid blocking the API
        thread = threading.Thread(target=run_online_sgd_update, daemon=True)
        thread.start()

def background_online_learning_loop():
    """Background task that checks every minute if online learning should run."""
    while True:
        try:
            check_and_run_online_learning()
        except Exception as e:
            print(f"[ONLINE-LEARNING] Error in background loop: {e}")
        
        # Sleep for 1 minute before checking again
        time.sleep(60)

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    """Start background online learning task on app startup."""
    print("[ONLINE-LEARNING] Starting background online learning task...")
    print("[ONLINE-LEARNING] Will run every 15 minutes OR every 67 new feedback samples")
    
    # Initialize feedback count
    _online_learning_state["last_feedback_count"] = count_feedback_samples()
    _online_learning_state["last_update_time"] = datetime.now()
    
    # Start background thread
    thread = threading.Thread(target=background_online_learning_loop, daemon=True)
    thread.start()
    print("[ONLINE-LEARNING] Background task started.")


# ------------------------
# Run server
# ------------------------

if __name__ == '__main__':
    import uvicorn
    
    MODEL_SERVER_PORT = 5000  # Port expected by client.py
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║   Credit Card Fraud Detection - Model Server             ║
    ╠══════════════════════════════════════════════════════════╣
    ║   Server running on: http://localhost:{MODEL_SERVER_PORT}                ║
    ║   API docs: http://localhost:{MODEL_SERVER_PORT}/docs                   ║
    ║                                                          ║
    ║   Endpoints:                                             ║
    ║   - POST /predict  - Make fraud predictions              ║
    ║   - POST /feedback - Submit feedback for online learning ║
    ║   - GET  /health   - Health check                        ║
    ║                                                          ║
    ║   Online Learning:                                       ║
    ║   - Runs every 15 minutes OR every 67 new samples       ║
    ║   - Model auto-reloads after update                     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host='0.0.0.0', port=MODEL_SERVER_PORT, reload=False)