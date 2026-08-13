
# For data manipulation
import pandas as pd

# For preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# For model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# For model serialization and experiment tracking
import joblib
import mlflow


# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Visit_With_Us_Tourism_Prediction")


# Xtrain/Xtest/ytrain/ytest are downloaded from the previous
# GitHub Actions job's artifact

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

print("Training data loaded successfully.")
print("Training shape:", Xtrain.shape)
print("Testing shape:", Xtest.shape)


# Define numerical and categorical features

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]


# Handle class imbalance

class_weight = (
    ytrain.value_counts()[0] /
    ytrain.value_counts()[1]
)

print("Class imbalance ratio:", class_weight)


# Define preprocessing steps

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# Define XGBoost model

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)


# Hyperparameter grid

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0]
}


# Model pipeline

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# Start MLflow run

with mlflow.start_run():

    # Hyperparameter tuning
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)

            mlflow.log_metric(
                "mean_test_score",
                mean_score
            )

            mlflow.log_metric(
                "std_test_score",
                std_score
            )

    print(
        f"Logged {len(results['params'])} "
        "hyperparameter combinations to MLflow."
    )


    # Log best parameters
    mlflow.log_params(
        grid_search.best_params_
    )

    print(
        "Best parameters:",
        grid_search.best_params_
    )

    print(
        "Best cross-validation F1-score:",
        grid_search.best_score_
    )


    # Store best model
    best_model = grid_search.best_estimator_


    # Classification threshold
    classification_threshold = 0.45


    # Training predictions
    y_pred_train_proba = (
        best_model.predict_proba(Xtrain)[:, 1]
    )

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)


    # Testing predictions
    y_pred_test_proba = (
        best_model.predict_proba(Xtest)[:, 1]
    )

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)


    # Classification reports
    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )


    # Print test report
    print("\n========== TEST CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            ytest,
            y_pred_test
        )
    )


    # Log metrics
    mlflow.log_metrics({

        "train_accuracy":
            train_report["accuracy"],

        "train_precision":
            train_report["1"]["precision"],

        "train_recall":
            train_report["1"]["recall"],

        "train_f1-score":
            train_report["1"]["f1-score"],

        "test_accuracy":
            test_report["accuracy"],

        "test_precision":
            test_report["1"]["precision"],

        "test_recall":
            test_report["1"]["recall"],

        "test_f1-score":
            test_report["1"]["f1-score"]
    })


    # Save trained model
    model_path = (
        "tourism_project/deployment/best_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # Log model as MLflow artifact
    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(
        f"Model saved to {model_path}"
    )
