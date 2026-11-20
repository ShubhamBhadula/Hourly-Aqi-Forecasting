# Hourly AQI Prediction and Analysis Using Apache Spark
# Four Spark MLlib regression models + persistence baseline + visualizations
# Run inside an existing PySpark shell:
# exec(open(r"E:\sprakproject\aqi_model_comparison_with_graphs.py").read())

import os
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator

# --------------------------- CONFIGURATION ---------------------------
INPUT_CSV = r"E:\sprakproject\aqi_hourly_features_2025.csv"
OUTPUT_DIR = r"E:\sprakproject\graphs"

FEATURES = [
    "aqi_now",
    "aqi_1h_ago",
    "aqi_2h_ago",
    "aqi_3h_ago",
    "aqi_24h_ago",
    "target_hour",
    "target_day_of_week",
    "target_month",
]
LABEL = "aqi_next_hour"
TIME_COLUMN = "forecast_origin_timestamp"
TARGET_TIME_COLUMN = "target_timestamp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib is only used to save the requested project graphs.
try:
    import matplotlib
    matplotlib.use("Agg")  # Save PNGs without opening interactive windows.
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "Matplotlib is required to create the project graphs. "
        "Install it in the Python environment used by PySpark with: pip install matplotlib"
    ) from exc

# --------------------------- LOAD DATA -------------------------------
print("\n========== DATASET ==========")
data = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)

required_columns = [TIME_COLUMN, TARGET_TIME_COLUMN] + FEATURES + [LABEL]
missing_columns = [c for c in required_columns if c not in data.columns]
if missing_columns:
    raise ValueError("Missing required dataset columns: " + ", ".join(missing_columns))

# Retain only columns required for training, testing, and prediction display.
data = data.select(*required_columns)
data = data.withColumn(TIME_COLUMN, F.to_timestamp(F.col(TIME_COLUMN)))
data = data.withColumn(TARGET_TIME_COLUMN, F.to_timestamp(F.col(TARGET_TIME_COLUMN)))
data = data.dropna(subset=required_columns)
data = data.orderBy(F.col(TIME_COLUMN).asc()).cache()

usable_rows = data.count()
if usable_rows < 10:
    raise ValueError("Not enough usable rows to train and test the models.")

train_count = int(usable_rows * 0.80)
# Chronological split: earliest 80% for training, latest 20% for testing.
train_raw = data.orderBy(F.col(TIME_COLUMN).asc()).limit(train_count).cache()
test_raw = (
    data.orderBy(F.col(TIME_COLUMN).desc())
        .limit(usable_rows - train_count)
        .orderBy(F.col(TIME_COLUMN).asc())
        .cache()
)

training_rows = train_raw.count()
testing_rows = test_raw.count()

print("Total usable rows :", usable_rows)
print("Training rows     :", training_rows)
print("Testing rows      :", testing_rows)
print("Forecast target   : AQI one hour after each forecast origin")
print("=============================\n")

# --------------------------- FEATURE ASSEMBLY ------------------------
assembler = VectorAssembler(
    inputCols=FEATURES,
    outputCol="features",
    handleInvalid="skip",
)
train = assembler.transform(train_raw).select("features", LABEL).cache()
test = assembler.transform(test_raw).select(
    TIME_COLUMN, TARGET_TIME_COLUMN, "aqi_now", LABEL, "features"
).cache()

# --------------------------- MODELS ----------------------------------
models = [
    ("Linear Regression", LinearRegression(featuresCol="features", labelCol=LABEL)),
    ("Decision Tree", DecisionTreeRegressor(featuresCol="features", labelCol=LABEL)),
    ("Random Forest", RandomForestRegressor(featuresCol="features", labelCol=LABEL)),
    ("Gradient-Boosted Trees", GBTRegressor(featuresCol="features", labelCol=LABEL)),
]

evaluators = {
    "MAE": RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="mae"),
    "RMSE": RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse"),
    "R2": RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="r2"),
}

ml_results = []
model_predictions = {}

for model_name, estimator in models:
    print("Training", model_name, "...")
    fitted_model = estimator.fit(train)
    prediction_df = fitted_model.transform(test).cache()

    mae = evaluators["MAE"].evaluate(prediction_df)
    rmse = evaluators["RMSE"].evaluate(prediction_df)
    r2 = evaluators["R2"].evaluate(prediction_df)

    ml_results.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2})
    model_predictions[model_name] = prediction_df

# --------------------------- PERSISTENCE BASELINE --------------------
# Benchmark: assume the next-hour AQI remains equal to the current AQI.
baseline_df = test_raw.select(
    F.col(TIME_COLUMN),
    F.col(TARGET_TIME_COLUMN),
    F.col("aqi_now"),
    F.col(LABEL),
    F.col("aqi_now").cast("double").alias("prediction"),
)

baseline_result = {
    "Model": "Persistence Baseline",
    "MAE": evaluators["MAE"].evaluate(baseline_df),
    "RMSE": evaluators["RMSE"].evaluate(baseline_df),
    "R2": evaluators["R2"].evaluate(baseline_df),
}

all_results = ml_results + [baseline_result]
best_result = min(ml_results, key=lambda item: item["RMSE"])
best_name = best_result["Model"]
best_predictions = model_predictions[best_name]

# --------------------------- PRINT RESULTS ----------------------------
print("\n============= MODEL COMPARISON =============")
print("{:<27s} {:>9s} {:>10s} {:>10s}".format("Model", "MAE", "RMSE", "R2"))
print("-" * 60)
for result in all_results:
    print("{:<27s} {:>9.3f} {:>10.3f} {:>10.3f}".format(
        result["Model"], result["MAE"], result["RMSE"], result["R2"]
    ))
print("=" * 60)
print("BEST ML MODEL:", best_name)
print("Lowest test RMSE:", round(best_result["RMSE"], 3))

# --------------------------- ACTUAL PREDICTIONS -----------------------
print("\n===== AQI PREDICTIONS: BEST MODEL =====")
print("Model:", best_name)
print("Forecast meaning: predict AQI at target_timestamp using earlier AQI readings.")
print("Showing the first 10 predictions from the test period:")

prediction_output = best_predictions.select(
    F.col(TIME_COLUMN).alias("Forecast origin"),
    F.col(TARGET_TIME_COLUMN).alias("Time being predicted"),
    F.col("aqi_now").alias("Current AQI"),
    F.col(LABEL).alias("Actual next-hour AQI"),
    F.round(F.col("prediction"), 2).alias("Predicted next-hour AQI"),
    F.round(F.abs(F.col(LABEL) - F.col("prediction")), 2).alias("Absolute error"),
).orderBy(F.col("Forecast origin").asc())

prediction_output.show(10, truncate=False)

# Collect only the held-out test predictions (1,663 rows in the current dataset)
# to create charts locally. This does not collect the training dataset.
chart_rows = (
    best_predictions.select(
        TIME_COLUMN,
        TARGET_TIME_COLUMN,
        F.col(LABEL).cast("double").alias("actual"),
        F.col("prediction").cast("double").alias("predicted"),
    )
    .orderBy(F.col(TIME_COLUMN).asc())
    .collect()
)

if not chart_rows:
    raise RuntimeError("No test predictions were available for visualization.")

forecast_times = [row[TARGET_TIME_COLUMN] for row in chart_rows]
actual_values = [row["actual"] for row in chart_rows]
predicted_values = [row["predicted"] for row in chart_rows]
errors = [actual - predicted for actual, predicted in zip(actual_values, predicted_values)]

# --------------------------- GRAPH 1: MODEL RMSE ----------------------
plt.figure(figsize=(10, 6))
labels = [r["Model"] for r in all_results]
rmse_values = [r["RMSE"] for r in all_results]
bars = plt.bar(labels, rmse_values)
plt.title("Test RMSE Comparison of AQI Forecasting Models")
plt.ylabel("Root Mean Squared Error (AQI points)")
plt.xlabel("Model")
plt.xticks(rotation=18, ha="right")
plt.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_model_rmse_comparison.png"), dpi=180)
plt.close()

# --------------------------- GRAPH 2: ACTUAL VS PREDICTED -------------
# Plot first 200 chronological test records for legibility.
plot_count = min(200, len(forecast_times))
plt.figure(figsize=(13, 6))
plt.plot(forecast_times[:plot_count], actual_values[:plot_count], label="Actual AQI", linewidth=1.8)
plt.plot(forecast_times[:plot_count], predicted_values[:plot_count], label="Predicted AQI", linewidth=1.5)
plt.title(f"Actual vs Predicted Next-Hour AQI ({plot_count} Test Predictions)")
plt.xlabel("Time being predicted")
plt.ylabel("AQI")
plt.legend()
plt.grid(alpha=0.25)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_actual_vs_predicted_aqi.png"), dpi=180)
plt.close()

# --------------------------- GRAPH 3: PREDICTION ERRORS ---------------
plt.figure(figsize=(13, 5))
plt.plot(forecast_times[:plot_count], errors[:plot_count], linewidth=1.4, label="Actual - Predicted")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title(f"Next-Hour AQI Prediction Error ({plot_count} Test Predictions)")
plt.xlabel("Time being predicted")
plt.ylabel("Prediction error (AQI points)")
plt.legend()
plt.grid(alpha=0.25)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_prediction_error.png"), dpi=180)
plt.close()

# --------------------------- GRAPH 4: WORKFLOW DIAGRAM ----------------
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")
workflow = [
    (0.3, "CPCB hourly\nAQI dataset"),
    (3.0, "Preprocessing\nand cleaning"),
    (5.7, "Feature extraction\n(current + lag AQI\n+ time features)"),
    (8.4, "Apache Spark\nMLlib models"),
    (11.1, "Evaluation and\nnext-hour AQI\nprediction"),
]
for x, label in workflow:
    ax.text(
        x + 1.05, 2.7, label,
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="white", edgecolor="black", linewidth=1.4),
    )
for i in range(len(workflow) - 1):
    x1 = workflow[i][0] + 2.15
    x2 = workflow[i + 1][0] - 0.08
    ax.annotate("", xy=(x2, 2.7), xytext=(x1, 2.7), arrowprops=dict(arrowstyle="->", linewidth=1.6))
ax.text(7, 0.8, "Chronological 80% training / 20% testing  |  MAE, RMSE and R²  |  Persistence baseline comparison",
        ha="center", va="center", fontsize=10)
ax.set_title("Hourly AQI Prediction Project Workflow", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_project_workflow_diagram.png"), dpi=180, bbox_inches="tight")
plt.close()

# --------------------------- SAVE PREDICTION CSV ----------------------
# Save the full best-model test prediction table for later report analysis.
prediction_output.coalesce(1).write.mode("overwrite").option("header", True).csv(
    os.path.join(OUTPUT_DIR, "best_model_test_predictions_csv")
)

print("\n========== VISUAL OUTPUTS CREATED ==========")
for filename in [
    "01_model_rmse_comparison.png",
    "02_actual_vs_predicted_aqi.png",
    "03_prediction_error.png",
    "04_project_workflow_diagram.png",
]:
    print(os.path.join(OUTPUT_DIR, filename))
print(os.path.join(OUTPUT_DIR, "best_model_test_predictions_csv"))
print("============================================")
print("NOTE: These are historical test-period predictions, not a live AQI forecast.")
print("The charts use actual held-out test predictions from the selected MLlib model.")

# Release cached DataFrames; keep the interactive Spark session running.
train.unpersist()
test.unpersist()
train_raw.unpersist()
test_raw.unpersist()
data.unpersist()
for prediction_df in model_predictions.values():
    prediction_df.unpersist()
