
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "creditguard-dev-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "creditguard_pipeline.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "input_features.pkl")
DB_PATH = os.path.join(BASE_DIR, "creditguard.db")

# Load trained model and its exact feature order
model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

EXPECTED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
]

if FEATURES != EXPECTED_FEATURES:
    raise ValueError(
        "Feature mismatch. Ensure the 12-feature model and input_features.pkl "
        "are from the same training run."
    )

NUMERIC_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

CATEGORICAL_OPTIONS = {
    "NAME_INCOME_TYPE": [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
    "NAME_FAMILY_STATUS": [
        "Single / not married",
        "Married",
        "Civil marriage",
        "Separated",
        "Widow",
        "Unknown",
    ],
}


def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                income REAL NOT NULL,
                credit REAL NOT NULL,
                probability REAL NOT NULL,
                risk_level TEXT NOT NULL
            )
        """)


init_db()


@app.route("/")
def home():
    return render_template(
        "index.html",
        features=FEATURES,
        numeric_features=NUMERIC_FEATURES,
        categorical_options=CATEGORICAL_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {}

        # Read numeric fields; blank external scores are imputed by the pipeline
        for feature in NUMERIC_FEATURES:
            raw_value = request.form.get(feature, "").strip()

            if raw_value == "":
                if feature.startswith("EXT_SOURCE_"):
                    values[feature] = np.nan
                    continue
                raise ValueError(f"Please enter {feature.replace('_', ' ').title()}.")

            value = float(raw_value)

            if not np.isfinite(value):
                raise ValueError(f"{feature} must be a valid number.")

            values[feature] = value

        # Basic input validation
        if values["AMT_INCOME_TOTAL"] <= 0:
            raise ValueError("Income must be greater than zero.")

        if values["AMT_CREDIT"] <= 0:
            raise ValueError("Credit amount must be greater than zero.")

        if values["AMT_ANNUITY"] <= 0:
            raise ValueError("Annuity must be greater than zero.")

        if values["AMT_GOODS_PRICE"] < 0:
            raise ValueError("Goods price cannot be negative.")

        for feature in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            score = values[feature]
            if not np.isnan(score) and not 0 <= score <= 1:
                raise ValueError(f"{feature} must be between 0 and 1.")

        for feature, options in CATEGORICAL_OPTIONS.items():
            value = request.form.get(feature, "").strip()

            if value not in options:
                raise ValueError(f"Please select a valid {feature.replace('_', ' ').title()}.")

            values[feature] = value

        # Match the exact training feature order
        input_df = pd.DataFrame(
            [[values[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        probability = float(model.predict_proba(input_df)[0][1])
        probability_pct = round(probability * 100, 2)

        # Demonstration thresholds only; not suitable for actual lending decisions
        if probability < 0.25:
            risk_level = "Low"
        elif probability < 0.50:
            risk_level = "Medium"
        else:
            risk_level = "High"

        with get_db() as connection:
            cursor = connection.execute("""
                INSERT INTO prediction_history
                (created_at, income, credit, probability, risk_level)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                values["AMT_INCOME_TOTAL"],
                values["AMT_CREDIT"],
                probability_pct,
                risk_level,
            ))
            prediction_id = cursor.lastrowid

        return render_template(
            "result.html",
            probability=probability_pct,
            risk_level=risk_level,
            income=values["AMT_INCOME_TOTAL"],
            credit=values["AMT_CREDIT"],
            prediction_id=prediction_id,
        )

    except (ValueError, TypeError) as error:
        flash(str(error), "error")
        return redirect(url_for("home"))


@app.route("/history")
def history():
    with get_db() as connection:
        records = connection.execute("""
            SELECT * FROM prediction_history
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return render_template("history.html", records=records)


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(FEATURES),
    }



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
