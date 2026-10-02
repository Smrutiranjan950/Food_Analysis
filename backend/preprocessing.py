import cv2
import numpy as np
import os


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def validate_image(image_path):
    """
    Check whether the image exists and can be read.
    """

    if not os.path.exists(image_path):
        return False, "Image file does not exist."

    extension = os.path.splitext(image_path)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return False, "Unsupported image format."

    image = cv2.imread(image_path)

    if image is None:
        return False, "Invalid or corrupted image."

    return True, "Image is valid."


# --------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------

def load_image(image_path):
    """
    Load an image using OpenCV.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load image.")

    return image


# --------------------------------------------------
# IMAGE RESIZING
# --------------------------------------------------

def resize_image(image, size=(224, 224)):
    """
    Resize image to the standard ML input size.
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


# --------------------------------------------------
# IMAGE NORMALIZATION
# --------------------------------------------------

def normalize_image(image):
    """
    Normalize pixel values from 0-255 to 0-1.
    """

    image = image.astype(np.float32)

    image = image / 255.0

    return image


# --------------------------------------------------
# COLOR FEATURE EXTRACTION
# --------------------------------------------------

def extract_color_features(image):
    """
    Extract average color information.
    """

    mean_bgr = np.mean(
        image,
        axis=(0, 1)
    )

    blue = float(mean_bgr[0])
    green = float(mean_bgr[1])
    red = float(mean_bgr[2])

    return {
        "average_blue": round(blue, 2),
        "average_green": round(green, 2),
        "average_red": round(red, 2)
    }


# --------------------------------------------------
# HSV FEATURES
# --------------------------------------------------

def extract_hsv_features(image):
    """
    Convert image to HSV and extract
    average Hue, Saturation and Value.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    mean_hsv = np.mean(
        hsv,
        axis=(0, 1)
    )

    return {
        "average_hue": round(float(mean_hsv[0]), 2),
        "average_saturation": round(float(mean_hsv[1]), 2),
        "average_value": round(float(mean_hsv[2]), 2)
    }


# --------------------------------------------------
# GRAYSCALE FEATURES
# --------------------------------------------------

def extract_texture_features(image):
    """
    Extract simple grayscale statistics
    for texture analysis.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "gray_mean": round(
            float(np.mean(gray)),
            2
        ),
        "gray_std": round(
            float(np.std(gray)),
            2
        )
    }


# --------------------------------------------------
# COMPLETE FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(image):
    """
    Extract color, HSV and texture features.
    """

    color_features = extract_color_features(
        image
    )

    hsv_features = extract_hsv_features(
        image
    )

    texture_features = extract_texture_features(
        image
    )

    features = {}

    features.update(color_features)
    features.update(hsv_features)
    features.update(texture_features)

    return features


# --------------------------------------------------
# COMPLETE PREPROCESSING PIPELINE
# --------------------------------------------------

def preprocess_image(image_path):
    """
    Complete image preprocessing pipeline.
    """

    valid, message = validate_image(
        image_path
    )

    if not valid:
        raise ValueError(message)

    image = load_image(
        image_path
    )

    resized = resize_image(
        image
    )

    normalized = normalize_image(
        resized
    )

    features = extract_features(
        resized
    )

    return {
        "original": image,
        "resized": resized,
        "normalized": normalized,
        "features": features
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("----------------------------------------")
    print("     FOOD IMAGE PREPROCESSING")
    print("----------------------------------------")

    print("Module loaded successfully.")

    print("\nFunctions available:")
    print("1. validate_image()")
    print("2. load_image()")
    print("3. resize_image()")
    print("4. normalize_image()")
    print("5. extract_color_features()")
    print("6. extract_hsv_features()")
    print("7. extract_texture_features()")
    print("8. extract_features()")
    print("9. preprocess_image()")

    print("\nOpenCV version:")
    print(cv2.__version__)

    print("----------------------------------------")