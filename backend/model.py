import os
import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from backend.preprocessing import preprocess_image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "images"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "food_model.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "food_dataset.csv"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "average_blue",
    "average_green",
    "average_red",
    "average_hue",
    "average_saturation",
    "average_value",
    "gray_mean",
    "gray_std"
]


# ============================================================
# LOAD IMAGE DATASET
# ============================================================

def load_dataset():

    features = []
    labels = []

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image dataset folder not found: {IMAGE_DIR}"
        )

    for class_name in sorted(
        os.listdir(IMAGE_DIR)
    ):

        class_path = os.path.join(
            IMAGE_DIR,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        print(
            f"Loading class: {class_name}"
        )

        for filename in os.listdir(
            class_path
        ):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                class_path,
                filename
            )

            try:

                result = preprocess_image(
                    image_path
                )

                image_features = result[
                    "features"
                ]

                feature_vector = [
                    image_features[name]
                    for name in FEATURE_NAMES
                ]

                features.append(
                    feature_vector
                )

                labels.append(
                    class_name
                )

            except Exception as e:

                print(
                    f"Skipping {image_path}: {e}"
                )

    if not features:
        raise ValueError(
            "No valid images found in dataset."
        )

    return (
        np.array(features),
        np.array(labels)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print()
    print("========================================")
    print("       FOOD CLASSIFICATION MODEL")
    print("========================================")

    X, y = load_dataset()

    print()
    print(
        f"Total images: {len(X)}"
    )

    print(
        f"Number of classes: {len(np.unique(y))}"
    )

    print(
        f"Classes: {sorted(np.unique(y))}"
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print()
    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_data = {
        "model": model,
        "classes": sorted(
            np.unique(y)
        ),
        "feature_names": FEATURE_NAMES
    }

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model_data,
            file
        )

    print()
    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print()
    print("========================================")
    print("       MODEL TRAINING COMPLETE")
    print("========================================")

    return model_data


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model_data = pickle.load(
            file
        )

    return model_data


# ============================================================
# PREDICT FOOD
# ============================================================

def predict_image(image_path):

    try:

        # Load saved model
        model_data = load_model()

        model = model_data[
            "model"
        ]

        feature_names = model_data[
            "feature_names"
        ]

        # Preprocess image
        result = preprocess_image(
            image_path
        )

        image_features = result[
            "features"
        ]

        # Create feature vector
        feature_vector = [
            image_features[name]
            for name in feature_names
        ]

        feature_array = np.array(
            [feature_vector]
        )

        # Prediction
        prediction = model.predict(
            feature_array
        )

        food_name = str(
            prediction[0]
        )

        # Confidence
        confidence = 0.0

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    feature_array
                )[0]
            )

            confidence = (
                float(
                    np.max(
                        probabilities
                    )
                ) * 100
            )

        return {
            "success": True,
            "food_name": food_name,
            "confidence": round(
                confidence,
                2
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# GET NUTRITION INFORMATION
# ============================================================

def get_food_info(food_name):

    if not os.path.exists(
        DATASET_PATH
    ):

        return None

    try:

        data = pd.read_csv(
            DATASET_PATH
        )

        if "food_name" not in data.columns:
            return None

        matches = data[
            data["food_name"]
            .astype(str)
            .str.lower()
            == str(food_name).lower()
        ]

        if matches.empty:
            return None

        row = matches.iloc[0]

        return row.to_dict()

    except Exception as e:

        print(
            f"Nutrition lookup error: {e}"
        )

        return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()