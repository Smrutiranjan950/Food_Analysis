import os
import cv2
import pickle
import warnings
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_image


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
# EXTRACT FEATURES FROM IMAGE
# ============================================================

def extract_image_features(image_path):
    """
    Preprocess image and extract visual features.
    """

    result = preprocess_image(
        image_path
    )

    features = result["features"]

    return [
        features["average_blue"],
        features["average_green"],
        features["average_red"],
        features["average_hue"],
        features["average_saturation"],
        features["average_value"],
        features["gray_mean"],
        features["gray_std"]
    ]


# ============================================================
# LOAD DATASET IMAGES
# ============================================================

def load_training_data():

    X = []
    y = []

    if not os.path.exists(IMAGE_DIR):

        print(
            "ERROR: Image directory not found:"
        )

        print(
            IMAGE_DIR
        )

        return X, y

    print(
        "\nLoading training images..."
    )

    food_classes = sorted(
        [
            folder
            for folder in os.listdir(
                IMAGE_DIR
            )
            if os.path.isdir(
                os.path.join(
                    IMAGE_DIR,
                    folder
                )
            )
        ]
    )

    for food_class in food_classes:

        class_path = os.path.join(
            IMAGE_DIR,
            food_class
        )

        image_files = [
            file
            for file in os.listdir(
                class_path
            )
            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]

        print(
            f"{food_class}: "
            f"{len(image_files)} images"
        )

        for image_file in image_files:

            image_path = os.path.join(
                class_path,
                image_file
            )

            try:

                features = extract_image_features(
                    image_path
                )

                X.append(
                    features
                )

                y.append(
                    food_class
                )

            except Exception as e:

                print(
                    f"Skipped {image_file}: {e}"
                )

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print(
        "\n========================================"
    )

    print(
        "       FOOD CLASSIFICATION MODEL"
    )

    print(
        "========================================"
    )

    X, y = load_training_data()

    if len(X) == 0:

        print(
            "\nERROR: No training images found."
        )

        return

    print(
        f"\nImages found: {len(X)}"
    )

    classes = sorted(
        list(
            set(y)
        )
    )

    print(
        f"Food classes: {len(classes)}"
    )

    for food_class in classes:

        count = y.count(
            food_class
        )

        print(
            f"{food_class}: "
            f"{count} images"
        )

    print(
        "\nCreating training/testing split..."
    )

    # --------------------------------------------------------
    # Check whether stratified split is possible
    # --------------------------------------------------------

    class_counts = {
        food_class: y.count(food_class)
        for food_class in classes
    }

    minimum_images = min(
        class_counts.values()
    )

    if minimum_images >= 5:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print(
            f"Training images: {len(X_train)}"
        )

        print(
            f"Testing images: {len(X_test)}"
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

        print(
            "\nTraining Random Forest model..."
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        print(
            "\n========================================"
        )

        print(
            f"Model Accuracy: "
            f"{accuracy * 100:.2f} %"
        )

        print(
            "========================================"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

    else:

        warnings.warn(
            "Some classes contain fewer than "
            "5 images. Training on all images."
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

        print(
            "\nTraining Random Forest model..."
        )

        model.fit(
            X,
            y
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_data = {
        "model": model,
        "classes": classes,
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

    print(
        "\n========================================"
    )

    print(
        "Model saved successfully!"
    )

    print(
        MODEL_PATH
    )

    print(
        "========================================"
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Trained model not found. "
            "Please train the model first."
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

        model_data = load_model()

        model = model_data["model"]

        classes = model_data["classes"]

        # Extract image features
        features = extract_image_features(
            image_path
        )

        # Convert to dataframe
        features_df = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )

        # Prediction
        prediction = model.predict(
            features_df
        )[0]

        # Probability
        probabilities = model.predict_proba(
            features_df
        )[0]

        max_probability = max(
            probabilities
        )

        confidence = round(
            float(max_probability) * 100,
            2
        )

        return {
            "success": True,
            "food_name": str(
                prediction
            ),
            "confidence": confidence
        }

    except Exception as e:

        return {
            "success": False,
            "food_name": None,
            "confidence": 0,
            "error": str(e)
        }


# ============================================================
# GET NUTRITION INFORMATION
# ============================================================

def get_food_info(food_name):

    """
    Get nutritional information from
    dataset/food_dataset.csv
    """

    if not os.path.exists(
        DATASET_PATH
    ):

        print(
            "Nutrition dataset not found:"
        )

        print(
            DATASET_PATH
        )

        return None

    try:

        df = pd.read_csv(
            DATASET_PATH
        )

        # Check required column
        if "food_name" not in df.columns:

            print(
                "ERROR: food_name column "
                "not found in dataset."
            )

            return None

        # Case-insensitive search
        result = df[
            df["food_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(food_name)
            .strip()
            .lower()
        ]

        if result.empty:

            return None

        return result.iloc[0].to_dict()

    except Exception as e:

        print(
            "Error reading nutrition dataset:"
        )

        print(
            e
        )

        return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()