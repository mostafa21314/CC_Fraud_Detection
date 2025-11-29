# Fraud Detection API – Backend Service Documentation

## 1. Overview

This API exposes the **final trained credit card fraud detection model** as a REST service.

- **Backend**: Python, FastAPI, PyTorch  
- **Base URL (dev)**: `http://localhost:8000`  
- **Model**: MLP with focal + cost-sensitive loss, trained on 80% of the data, tuned with 5-fold CV and threshold tuning.  
- **Main capabilities**:
  - Predict fraud probability and binary label for a single transaction.
  - Log user feedback / ground truth labels for later **online retraining**.

Your web app will mainly use:

- `POST /predict` – to get a prediction  
- `POST /feedback` – to send back the true label and context after the prediction

---

## 2. Endpoints Summary

| Method | Path        | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/health`  | Health check, confirms service is running     |
| POST   | `/predict` | Predict fraud probability and label           |
| POST   | `/feedback`| Log feedback (true label) for online learning |

---

## 3. Features & Data Format

### 3.1 What the model expects

The model expects a **fixed set of numeric features** for each transaction.  
These are the columns from the processed dataset **excluding** the target column `Is Fraud?`.

The features are all **already-engineered / preprocessed** (normalized, z-scored, one-hot indicators, etc.).

The client (web app) must send them in a JSON object under `"features"` with:

- Keys = feature names (exact strings below)  
- Values = numeric (int/float)

> Ordering in JSON doesn’t matter – the backend will reorder internally.  
> But **all required keys must be present**.

### 3.2 Exact feature list

These are the **input features**:

```text
avg_transaction_per_user_norm,
avg_transaction_per_card_norm,
seconds_1990,
Amount_zscore,
OnlineFlag,
HasError,
DevCat_Developing,
DevCat_UnderDeveloped,
DevCat_Developed,
Amount_IsRefund,
MCC_1711,
MCC_3000,
MCC_3001,
MCC_3005,
MCC_3006,
MCC_3007,
MCC_3008,
MCC_3009,
MCC_3058,
MCC_3066,
MCC_3075,
MCC_3132,
MCC_3144,
MCC_3174,
MCC_3256,
MCC_3260,
MCC_3359,
MCC_3387,
MCC_3389,
MCC_3390,
MCC_3393,
MCC_3395,
MCC_3405,
MCC_3504,
MCC_3509,
MCC_3596,
MCC_3640,
MCC_3684,
MCC_3722,
MCC_3730,
MCC_3771,
MCC_3775,
MCC_3780,
MCC_4111,
MCC_4112,
MCC_4121,
MCC_4131,
MCC_4214,
MCC_4411,
MCC_4511,
MCC_4722,
MCC_4784,
MCC_4814,
MCC_4829,
MCC_4899,
MCC_4900,
MCC_5045,
MCC_5094,
MCC_5192,
MCC_5193,
MCC_5211,
MCC_5251,
MCC_5261,
MCC_5300,
MCC_5310,
MCC_5311,
MCC_5411,
MCC_5499,
MCC_5533,
MCC_5541,
MCC_5621,
MCC_5651,
MCC_5655,
MCC_5661,
MCC_5712,
MCC_5719,
MCC_5722,
MCC_5732,
MCC_5733,
MCC_5812,
MCC_5813,
MCC_5814,
MCC_5815,
MCC_5816,
MCC_5912,
MCC_5921,
MCC_5932,
MCC_5941,
MCC_5942,
MCC_5947,
MCC_5970,
MCC_5977,
MCC_6300,
MCC_7011,
MCC_7210,
MCC_7230,
MCC_7276,
MCC_7349,
MCC_7393,
MCC_7531,
MCC_7538,
MCC_7542,
MCC_7549,
MCC_7801,
MCC_7802,
MCC_7832,
MCC_7922,
MCC_7995,
MCC_7996,
MCC_8011,
MCC_8021,
MCC_8041,
MCC_8043,
MCC_8049,
MCC_8062,
MCC_8099,
MCC_8111,
MCC_8931,
MCC_9402


```
## 4. Endpoint Details

### 4.1 GET /health
**Description:** Check if the service is running.

**Request:**  
- **Method:** GET  
- **URL:** `/health`  
- **Body:** none  

**Example Response (200):**
```json
{
  "status": "ok",
  "detail": "Fraud model service is running."
}
```
---

### 4.2 POST /predict
**Description:** Predict fraud probability and binary label for one transaction.

**Request:**  
- **Method:** POST  
- **URL:** `/predict`  
- **Headers:** `Content-Type: application/json`

**Body:**
```json
{
  "features": {
    "avg_transaction_per_user_norm": 0.15,
    "avg_transaction_per_card_norm": 0.08,
    "seconds_1990": 1234567890.0,
    "Amount_zscore": 2.1,
    "OnlineFlag": 1,
    "HasError": 0,
    "DevCat_Developing": 1,
    "DevCat_UnderDeveloped": 0,
    "DevCat_Developed": 0,
    "Amount_IsRefund": 0,
    "MCC_1711": 0,
    "MCC_3000": 0,
    "MCC_3001": 0,
    "MCC_3005": 0,
    "...": 0,
    "MCC_8931": 0,
    "MCC_9402": 0
  }
}
```

All feature keys must be present.  
Values must be numeric.

**Response (200):**
```json
{
  "prob_not_fraud": 0.1268,
  "prob_fraud": 0.8732,
  "is_fraud": 1,
  "threshold": 0.52
}
```

**Error Cases:**
- `400 Bad Request` – missing/malformed feature  
- `500 Internal Server Error`

---

### 4.3 POST /feedback
**Description:**  
Log feedback (ground truth) to `artifacts/feedback_log.jsonl`.

**Request:**  
- **Method:** POST  
- **URL:** `/feedback`  
- **Headers:** `Content-Type: application/json`

**Body:**
```json
{
  "features": {
    "avg_transaction_per_user_norm": 0.15,
    "avg_transaction_per_card_norm": 0.08,
    "seconds_1990": 1234567890.0,
    "Amount_zscore": 2.1,
    "OnlineFlag": 1,
    "HasError": 0,
    "DevCat_Developing": 1,
    "DevCat_UnderDeveloped": 0,
    "DevCat_Developed": 0,
    "Amount_IsRefund": 0,
    "MCC_1711": 0,
    "...": 0,
    "MCC_8931": 0,
    "MCC_9402": 0
  },
  "model_prob_fraud": 0.8732,
  "model_prob_not_fraud": 0.1268,
  "model_label": 1,
  "true_label": 1,
  "metadata": {
    "transaction_id": "txn-001",
    "user_id": "user-42"
  }
}
```

**Response (200):**
```json
{
  "status": "ok",
  "message": "Feedback recorded successfully."
}
```

---

## 5. Example Client Calls

### 5.1 curl – Predict
```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{
    "features": {
      "avg_transaction_per_user_norm": 0.15,
      "avg_transaction_per_card_norm": 0.08,
      "seconds_1990": 1234567890.0,
      "Amount_zscore": 2.1,
      "OnlineFlag": 1,
      "HasError": 0,
      "DevCat_Developing": 1,
      "DevCat_UnderDeveloped": 0,
      "DevCat_Developed": 0,
      "Amount_IsRefund": 0,
      "MCC_1711": 0,
      "MCC_3000": 0,
      "MCC_3001": 0,
      "MCC_3005": 0,
      "MCC_8931": 0,
      "MCC_9402": 0
    }
  }'
```

### 5.2 curl – Feedback
```bash
curl -X POST "http://localhost:8000/feedback"   -H "Content-Type: application/json"   -d '{
    "features": {
      "avg_transaction_per_user_norm": 0.15,
      "avg_transaction_per_card_norm": 0.08,
      "seconds_1990": 1234567890.0,
      "Amount_zscore": 2.1,
      "OnlineFlag": 1,
      "HasError": 0,
      "DevCat_Developing": 1,
      "DevCat_UnderDeveloped": 0,
      "DevCat_Developed": 0,
      "Amount_IsRefund": 0,
      "MCC_1711": 0,
      "MCC_8931": 0,
      "MCC_9402": 0
    },
    "model_prob_fraud": 0.8732,
    "model_prob_not_fraud": 0.1268,
    "model_label": 1,
    "true_label": 0,
    "metadata": {
      "transaction_id": "txn-001",
      "user_id": "user-42"
    }
  }'
```

---

## 6. Running the Service

Inside the `ml_service` directory (with `venv` and `artifacts/final_model.pt`):

```bash
uvicorn api:app --reload
```

Then:  
Swagger UI: http://localhost:8000/docs  
Health: http://localhost:8000/health
