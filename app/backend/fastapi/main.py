import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Demand Planning & Forecasting API",
    description="API for market divergence predictions, segmenting markets, and forecasting multi-week Search Interest based on location and category lookups.",
    version="4.0.0"
)

# ------------------------------------------------------------------
# Paths Setup (Reliable Static Resolution)
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go 3 levels up from app/backend/fastapi/ to reach project root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

# Resolve Pickles Directory (Checks PROJECT_ROOT/pickles first, then PROJECT_ROOT/app/pickles)
PICKLE_DIR = os.path.join(PROJECT_ROOT, "pickles")
if not os.path.exists(PICKLE_DIR):
    PICKLE_DIR = os.path.join(PROJECT_ROOT, "app", "pickles")

# Resolve Data Directory (Checks PROJECT_ROOT/data/processed first, then PROJECT_ROOT/app/data/processed)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")

DIVERGENCE_CSV_PATH = os.path.join(DATA_DIR, "Demand_ot_hype_Divergence_score_features.csv")
CLUSTER_CSV_PATH = os.path.join(DATA_DIR, "market_segmentation.csv")
FORECAST_CSV_PATH = os.path.join(DATA_DIR, "Search_Intrest_forecasting_features.csv")

# Artifact Paths
XGB_MODEL_PATH = os.path.join(PICKLE_DIR, "xgb_divergence_model.pkl")
OHE_ENCODER_PATH = os.path.join(PICKLE_DIR, "onehot_encoder.pkl") 
REG_SCALER_PATH = os.path.join(PICKLE_DIR, "scaler.pkl")          
KMEANS_MODEL_PATH = os.path.join(PICKLE_DIR, "kmeans_model.pkl")
KMEANS_SCALER_PATH = os.path.join(PICKLE_DIR, "Kmeans_scaler.pkl")
LGB_FORECAST_MODEL_PATH = os.path.join(PICKLE_DIR, "lightgbm_search_Intrest_forecast_model.pkl")
LGB_FORECAST_META_PATH = os.path.join(PICKLE_DIR, "lightgbm_search_Intrest_forecast_model_metadata.pkl")

# ------------------------------------------------------------------
# Load Artifacts & Processed CSV Data (Isolated Blocks)
# ------------------------------------------------------------------
# 1. XGBoost Regression Artifacts
try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
    onehot_encoder = joblib.load(OHE_ENCODER_PATH)
    reg_scaler = joblib.load(REG_SCALER_PATH)
    print("✅ Regression artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Regression artifacts: {e}")
    xgb_model = onehot_encoder = reg_scaler = None

# 2. KMeans Clustering Artifacts
try:
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
    print("✅ Clustering artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Clustering artifacts: {e}")
    kmeans_model = kmeans_scaler = None

# 3. LightGBM Forecasting Artifacts
try:
    lgb_forecast_model = joblib.load(LGB_FORECAST_MODEL_PATH)
    lgb_forecast_metadata = joblib.load(LGB_FORECAST_META_PATH)
    print("✅ Forecasting artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Forecasting artifacts: {e}")
    lgb_forecast_model = lgb_forecast_metadata = None

# 4. Feature Datasets
try:
    df_divergence = pd.read_csv(DIVERGENCE_CSV_PATH)
    df_cluster = pd.read_csv(CLUSTER_CSV_PATH)
    df_forecast = pd.read_csv(FORECAST_CSV_PATH)
    print(f"✅ All CSV feature datasets successfully loaded from: {DATA_DIR}")
except Exception as e:
    print(f"❌ Error loading CSV files from {DATA_DIR}: {e}")
    df_divergence = df_cluster = df_forecast = None


# ------------------------------------------------------------------
# Simplified Pydantic Request Schemas
# ------------------------------------------------------------------
class BasicMarketInput(BaseModel):
    Country_Name: str = Field(..., examples=["United_States"])
    Category: str = Field(..., examples=["Fitness_Wearables"])


class ForecastInput(BaseModel):
    Country_Name: str = Field(..., examples=["India"])
    Category: str = Field(..., examples=["Fashion_Beauty"])
    Forecast_Horizon_Weeks: int = Field(default=4, ge=1, le=12, examples=[4])


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
# Endpoint 1: Regression (Divergence Score Prediction)
# ------------------------------------------------------------------
@app.post("/predict")
def predict_divergence(data: BasicMarketInput):
    if not all([xgb_model, onehot_encoder, reg_scaler]):
        raise HTTPException(status_code=500, detail="Regression pipeline artifacts are not fully loaded.")
    if df_divergence is None:
        raise HTTPException(status_code=500, detail="Divergence dataset is not loaded.")

    try:
        matching_rows = df_divergence[
            (df_divergence["Country_Name"] == data.Country_Name) & 
            (df_divergence["Category"] == data.Category)
        ]

        if matching_rows.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No matching data found for Country '{data.Country_Name}' and Category '{data.Category}'."
            )

        raw_df = matching_rows.iloc[[-1]].copy()

        cols_to_scale = [
            "Search_Velocity", "Search_Acceleration", "Media_Volume",
            "tone_net_sentiment", "tone_polarity", "tone_activity_density",
            "Source_Diversity", "Inflation_Rate", "Internet_Penetration",
            "holiday_count", "Search_Velocity_Lag1", "Search_Velocity_Lag2"
        ]
        categorical_cols = ["Country_Name", "Category"]

        scaled_nums = reg_scaler.transform(raw_df[cols_to_scale])
        scaled_df = pd.DataFrame(scaled_nums, columns=cols_to_scale, index=raw_df.index)

        encoded_cats = onehot_encoder.transform(raw_df[categorical_cols])
        encoded_cat_cols = onehot_encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=raw_df.index)

        processed_input = scaled_df.join(encoded_df)

        if hasattr(xgb_model, "feature_names_in_"):
            processed_input = processed_input.reindex(columns=xgb_model.feature_names_in_, fill_value=0.0)

        prediction = xgb_model.predict(processed_input)[0]

        return {
            "Country_Name": data.Country_Name,
            "Category": data.Category,
            "Divergence_Score_Prediction": float(np.round(prediction, 4)),
            "Interpretation": (
                "Under-served Opportunity (High Demand relative to Media Hype)"
                if prediction > 0
                else "Over-hyped Market (High Media Coverage relative to Demand)"
            ),
            "status": "success"
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Regression prediction error: {str(e)}")


# ------------------------------------------------------------------
# Endpoint 2: Clustering (Market Segmentation)
# ------------------------------------------------------------------
@app.post("/cluster")
def predict_cluster(data: BasicMarketInput):
    if not all([kmeans_model, kmeans_scaler]):
        raise HTTPException(status_code=500, detail="Clustering pipeline artifacts are not fully loaded.")
    if df_cluster is None:
        raise HTTPException(status_code=500, detail="Clustering dataset is not loaded.")

    try:
        matching_rows = df_cluster[
            (df_cluster["Country_Name"] == data.Country_Name) & 
            (df_cluster["Category"] == data.Category)
        ]

        if matching_rows.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No matching data found for Country '{data.Country_Name}' and Category '{data.Category}'."
            )

        raw_df = matching_rows.iloc[[-1]].copy()

        cluster_features = [
            "Mean_Search_Interest",
            "Mean_Media_Volume",
            "Mean_Demand_to_Hype_Ratio",
            "Search_Interest_Trend_Slope",
            "Mean_Net_Sentiment",
            "Mean_GDP_Per_Capita"
        ]

        scaled_input = kmeans_scaler.transform(raw_df[cluster_features])
        cluster_id = int(kmeans_model.predict(scaled_input)[0])

        cluster_info = CLUSTER_MAPPING.get(cluster_id, {
            "Name": "Unknown Segment",
            "Strategy": "No specific guidance available."
        })

        return {
            "Country_Name": data.Country_Name,
            "Category": data.Category,
            "Cluster_ID": cluster_id,
            "Segment_Name": cluster_info["Name"],
            "Strategic_Implication": cluster_info["Strategy"],
            "status": "success"
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Clustering error: {str(e)}")


# ------------------------------------------------------------------
# Endpoint 3: Time Series Search Interest Forecasting (LightGBM)
# ------------------------------------------------------------------
@app.post("/forecast")
def forecast_search_interest(data: ForecastInput):
    if not all([lgb_forecast_model, lgb_forecast_metadata]):
        raise HTTPException(status_code=500, detail="Forecasting model artifacts are not fully loaded.")
    if df_forecast is None:
        raise HTTPException(status_code=500, detail="Forecasting dataset is not loaded.")

    try:
        matching_rows = df_forecast[
            (df_forecast["Country_Name"] == data.Country_Name) & 
            (df_forecast["Category"] == data.Category)
        ].copy()

        if matching_rows.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No historical data found for Country '{data.Country_Name}' and Category '{data.Category}'."
            )

        date_col = 'Week_Start' if 'Week_Start' in matching_rows.columns else 'Date'
        matching_rows[date_col] = pd.to_datetime(matching_rows[date_col])
        series_df = matching_rows.sort_values(date_col).reset_index(drop=True)

        if len(series_df) < 4:
            raise HTTPException(
                status_code=400, 
                detail=f"Found only {len(series_df)} records for this combination. At least 4 historical points are required."
            )

        history_dates = list(series_df[date_col])
        history_values = list(series_df['Search_Interest'])
        
        forecast_dates = []
        forecast_values = []
        last_date = history_dates[-1]

        for w in range(1, data.Forecast_Horizon_Weeks + 1):
            next_date = last_date + pd.Timedelta(weeks=w)
            forecast_dates.append(next_date)
            
            combined_values = history_values + forecast_values
            
            feat_dict = {
                'Country_Name': data.Country_Name,
                'Category': data.Category,
                'Search_Interest_lag_1': combined_values[-1],
                'Search_Interest_lag_2': combined_values[-2] if len(combined_values) >= 2 else combined_values[-1],
                'Search_Interest_lag_4': combined_values[-4] if len(combined_values) >= 4 else combined_values[-1],
                'Search_Interest_roll_mean_4w': float(np.mean(combined_values[-4:])),
                'Search_Interest_roll_std_4w': float(np.std(combined_values[-4:])),
                'sin_week': float(np.sin(2 * np.pi * next_date.isocalendar().week / 52)),
                'cos_week': float(np.cos(2 * np.pi * next_date.isocalendar().week / 52))
            }

            for col in lgb_forecast_metadata['features']:
                if col not in feat_dict:
                    feat_dict[col] = 0.0

            input_df = pd.DataFrame([feat_dict])

            for col in lgb_forecast_metadata.get('categorical_features', []):
                if col in lgb_forecast_metadata.get('categorical_categories', {}):
                    cats = lgb_forecast_metadata['categorical_categories'][col]
                    input_df[col] = pd.Categorical(input_df[col], categories=cats)
                else:
                    input_df[col] = input_df[col].astype('category')

            input_df = input_df[lgb_forecast_metadata['features']]

            pred = float(lgb_forecast_model.predict(input_df)[0])
            pred_clamped = float(np.clip(pred, 0, 100))
            
            forecast_values.append(pred_clamped)

        forecast_results = [
            {
                "Forecast_Week": date.strftime('%Y-%m-%d'),
                "Predicted_Search_Interest": round(val, 2)
            }
            for date, val in zip(forecast_dates, forecast_values)
        ]

        return {
            "Country_Name": data.Country_Name,
            "Category": data.Category,
            "Forecast_Horizon_Weeks": data.Forecast_Horizon_Weeks,
            "Predictions": forecast_results,
            "status": "success"
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Forecasting error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)