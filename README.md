# ML-Based Intrusion Detection System (IDS) API

A machine learning–powered Intrusion Detection System (IDS).  
The API analyzes network flow features and classifies traffic as benign or malicious, with severity scoring and batch inference support.

---

## Model Overview

- **Dataset**: UNSW-NB15 (network intrusion dataset)
- **Task**: Binary classification (benign vs attack)
- **Model**: Logistic Regression
- **Pipeline**:
  - Categorical encoding
  - Numerical scaling
  - Missing-value handling
- **Output**:
  - Prediction
  - Confidence score
  - Severity level
  - Important contributing features

---
## How to Run With Docker

### 1. Build the image
```
docker build -t ml_ids .
```

### 2. Run the container
```
docker run -p 8000:8000 ml_ids
```

### 3. Open the Swagger UI
http://127.0.0.1:8000/

---

## How to Run Without Docker

### 1. Create virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
``` pip install -r requirements.txt```

### 3. Start the API
``` uvicorn src.serve_api:app --reload```

### 4. Open Swagger UI
http://127.0.0.1:8000/