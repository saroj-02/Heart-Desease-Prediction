import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):

    # -----------------------------
    # Remove duplicate rows
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # Convert target into binary
    # -----------------------------
    # num = 0  -> No Heart Disease
    # num > 0  -> Heart Disease

    df["target"] = df["num"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Remove original target column
    df = df.drop("num", axis=1)

    # Remove ID because it has no predictive meaning
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # dataset indicates hospital/dataset source.
    # It is not useful for our prediction.
    if "dataset" in df.columns:
        df = df.drop("dataset", axis=1)

    # -----------------------------
    # Separate features and target
    # -----------------------------

    X = df.drop("target", axis=1)
    y = df["target"]

    # -----------------------------
    # Identify column types
    # -----------------------------

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # -----------------------------
    # Numerical preprocessing
    # -----------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # -----------------------------
    # Categorical preprocessing
    # -----------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # -----------------------------
    # Combine preprocessing
    # -----------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # -----------------------------
    # Train/Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )