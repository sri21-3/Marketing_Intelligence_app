import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Demand-Hype Divergence & Segmentation API",
    description="API for predicting market divergence scores and segmenting country-category markets.",
    version="2.0.0"
)

# ------------------------------------------------------------------
# 1. Dynamically Load All Pickled Artifacts
# ------------------------------------------------------------------
# Get directory of the current script (e.g., app/backend/fastapi)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust this path depending on where your 'pickles' folder actually sits:
# Option A: 'pickles' is in the same folder as app1.py -> os.path.join(CURRENT_DIR, "pickles")
# Option B: 'pickles' is in the parent 'app' folder (2 levels up) -> os.path.abspath(os.path.join(CURRENT_DIR, "../../pickles"))
PICKLE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../pickles"))

# Fallback check if Option B directory doesn't exist, try local directory
if not os.path.exists(PICKLE_DIR):
    PICKLE_DIR = os.path.join(CURRENT_DIR, "pickles")

# Regression Artifacts
XGB_MODEL_PATH = os.path.join(PICKLE_DIR, "xgb_divergence_model.pkl")
OHE_ENCODER_PATH = os.path.join(PICKLE_DIR, "onehot_encoder.pkl") 
REG_SCALER_PATH = os.path.join(PICKLE_DIR, "scaler.pkl")          

# Clustering Artifacts
KMEANS_MODEL_PATH = os.path.join(PICKLE_DIR, "kmeans_model.pkl")
KMEANS_SCALER_PATH = os.path.join(PICKLE_DIR, "Kmeans_scaler.pkl")

try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
    onehot_encoder = joblib.load(OHE_ENCODER_PATH)
    reg_scaler = joblib.load(REG_SCALER_PATH)
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
    print(f"✅ All ML artifacts successfully loaded from: {PICKLE_DIR}")
except Exception as e:
    print(f"❌ Error loading pickle files from {PICKLE_DIR}: {e}")
    xgb_model = onehot_encoder = reg_scaler = kmeans_model = kmeans_scaler = None


# ------------------------------------------------------------------
# 2. Define Pydantic Schemas (Pydantic V2 Compliant)
# ------------------------------------------------------------------
class DivergenceInput(BaseModel):
    Country_Name: str = Field(..., examples=["United States"])
    Category: str = Field(..., examples=["Fitness_Wearables"])
    Search_Velocity: float = Field(..., examples=[12.5])
    Search_Acceleration: float = Field(..., examples=[1.2])
    Media_Volume: float = Field(..., examples=[450.0])
    tone_net_sentiment: float = Field(..., examples=[0.15])
    tone_polarity: float = Field(..., examples=[3.2])
    tone_activity_density: float = Field(..., examples=[0.85])
    Source_Diversity: float = Field(..., examples=[14.0])
    Inflation_Rate: float = Field(..., examples=[3.1])
    Internet_Penetration: float = Field(..., examples=[92.0])
    holiday_count: float = Field(..., examples=[1.0])
    Search_Velocity_Lag1: float = Field(..., examples=[10.0])
    Search_Velocity_Lag2: float = Field(..., examples=[8.5])


class ClusterInput(BaseModel):
    Mean_Search_Interest: float = Field(..., examples=[55.4])
    Mean_Media_Volume: float = Field(..., examples=[12000.0])
    Mean_Demand_to_Hype_Ratio: float = Field(..., examples=[0.045])
    Search_Interest_Trend_Slope: float = Field(..., examples=[0.12])
    Mean_Net_Sentiment: float = Field(..., examples=[1.85])
    Mean_GDP_Per_Capita: float = Field(..., examples=[45000.0])


# Business-Interpretable Map for Cluster Labels
CLUSTER_MAPPING = {
    0: {
        "Name": "Developing / Low-Hype Markets",
        "Strategy": "Watch & Seed: Low current saturation; ideal for early-stage expansion with lower media acquisition costs."
    },
    1: {
        "Name": "Over-Saturated & Noise-Heavy",
        "Strategy": "Re-Evaluate / Optimize: Organic reach is suppressed by media noise; requires hyper-targeted ad spend rather than broad campaigns."
    },
    2: {
        "Name": "Established High-Value Markets",
        "Strategy": "Core Investment / Monetize: Primary revenue drivers; high consumer intent coupled with strong willingness to pay."
    },
    3: {
        "Name": "High-Demand Emerging Markets",
        "Strategy": "Market Penetration: Strong baseline organic demand with low media competition; focus on brand awareness and localization."
    },
    4: {
        "Name": "Underserved High-Virality Markets",
        "Strategy": "Immediate Capture: Unmet consumer interest with minimal brand noise; high ROI opportunity for first-movers."
    }
}


# ------------------------------------------------------------------
# 3. Endpoint 1: Regression (Divergence Score Prediction)
# ------------------------------------------------------------------
@app.post("/predict")
def predict_divergence(data: DivergenceInput):
    if not all([xgb_model, onehot_encoder, reg_scaler]):
        raise HTTPException(status_code=500, detail="Regression pipeline artifacts are not fully loaded")

    try:
        raw_df = pd.DataFrame([data.model_dump()])

        cols_to_scale = [
            "Search_Velocity", "Search_Acceleration", "Media_Volume",
            "tone_net_sentiment", "tone_polarity", "tone_activity_density",
            "Source_Diversity", "Inflation_Rate", "Internet_Penetration",
            "holiday_count", "Search_Velocity_Lag1", "Search_Velocity_Lag2"
        ]
        categorical_cols = ["Country_Name", "Category"]

        # 1. Scale numeric features
        scaled_nums = reg_scaler.transform(raw_df[cols_to_scale])
        scaled_df = pd.DataFrame(scaled_nums, columns=cols_to_scale, index=raw_df.index)

        # 2. Encode categorical features
        encoded_cats = onehot_encoder.transform(raw_df[categorical_cols])
        encoded_cat_cols = onehot_encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=raw_df.index)

        # 3. Combine processed features
        processed_input = scaled_df.join(encoded_df)

        # 4. Enforce feature alignment expected by XGBoost
        if hasattr(xgb_model, "feature_names_in_"):
            processed_input = processed_input.reindex(columns=xgb_model.feature_names_in_, fill_value=0.0)

        # 5. Predict
        prediction = xgb_model.predict(processed_input)[0]

        return {
            "Divergence_Score_Prediction": float(np.round(prediction, 4)),
            "Interpretation": (
                "Under-served Opportunity (High Demand relative to Media Hype)"
                if prediction > 0
                else "Over-hyped Market (High Media Coverage relative to Demand)"
            ),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Regression prediction error: {str(e)}")


# ------------------------------------------------------------------
# 4. Endpoint 2: Clustering (Market Segmentation)
# ------------------------------------------------------------------
@app.post("/cluster")
def predict_cluster(data: ClusterInput):
    if not all([kmeans_model, kmeans_scaler]):
        raise HTTPException(status_code=500, detail="Clustering pipeline artifacts are not fully loaded")

    try:
        # Convert Pydantic model to DataFrame
        raw_df = pd.DataFrame([data.model_dump()])

        cluster_features = [
            "Mean_Search_Interest",
            "Mean_Media_Volume",
            "Mean_Demand_to_Hype_Ratio",
            "Search_Interest_Trend_Slope",
            "Mean_Net_Sentiment",
            "Mean_GDP_Per_Capita"
        ]

        # 1. Scale input features using Kmeans_scaler
        scaled_input = kmeans_scaler.transform(raw_df[cluster_features])

        # 2. Predict cluster ID using KMeans
        cluster_id = int(kmeans_model.predict(scaled_input)[0])

        # 3. Map cluster ID to business label and strategy
        cluster_info = CLUSTER_MAPPING.get(cluster_id, {
            "Name": "Unknown Segment",
            "Strategy": "No specific guidance available."
        })

        return {
            "Cluster_ID": cluster_id,
            "Segment_Name": cluster_info["Name"],
            "Strategic_Implication": cluster_info["Strategy"],
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Clustering error: {str(e)}")

# Fixed target module string to match 'app1:app'
if __name__ == "__main__":
    uvicorn.run("app1:app", host="127.0.0.1", port=8000, reload=True)