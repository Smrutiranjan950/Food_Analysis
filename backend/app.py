import os

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

from backend.preprocessing import preprocess_image
from backend.model import predict_image, get_food_info
from backend.database import (
    create_table,
    save_analysis,
    get_history,
    delete_history
)


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# DATABASE
# ============================================================

create_table()


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


def allowed_file(filename):
    """
    Check whether uploaded file has an allowed extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# ANALYZE FOOD IMAGE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # Check whether file exists in request
    if "food_image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["food_image"]

    # Check whether user selected a file
    if file.filename == "":
        return redirect(url_for("home"))

    # Check file extension
    if not allowed_file(file.filename):
        return (
            "Invalid image format. "
            "Please upload JPG, JPEG or PNG."
        )

    filename = file.filename

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        # Save uploaded image
        file.save(image_path)

        # ----------------------------------------------------
        # IMAGE PREPROCESSING
        # ----------------------------------------------------

        preprocessing_result = preprocess_image(
            image_path
        )

        features = preprocessing_result["features"]

        # ----------------------------------------------------
        # FOOD PREDICTION
        # ----------------------------------------------------

        prediction = predict_image(
            image_path
        )

        if not prediction["success"]:

            return (
                "Prediction failed: "
                + prediction.get(
                    "error",
                    "Unknown error"
                )
            )

        food_name = prediction["food_name"]
        confidence = prediction["confidence"]

        # ----------------------------------------------------
        # NUTRITION INFORMATION
        # ----------------------------------------------------

        food_info = get_food_info(
            food_name
        )

        # ----------------------------------------------------
        # SAVE ANALYSIS HISTORY
        # ----------------------------------------------------

        save_analysis(
            filename=filename,
            food_name=food_name,
            confidence=confidence,
            food_info=food_info
        )

        # ----------------------------------------------------
        # SHOW RESULT
        # ----------------------------------------------------

        return render_template(
            "result.html",
            filename=filename,
            food_name=food_name,
            confidence=confidence,
            food_info=food_info,
            features=features
        )

    except Exception as e:

        print("ERROR:", str(e))

        return (
            "Error while analyzing image: "
            + str(e)
        )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    history_data = get_history()

    return render_template(
        "history.html",
        history=history_data
    )


# ============================================================
# DELETE HISTORY
# ============================================================

@app.route(
    "/delete-history",
    methods=["POST"]
)
def delete_history_route():

    delete_history()

    return redirect(
        url_for("history")
    )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# FILE TOO LARGE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File is too large. "
        "Maximum allowed size is 16 MB.",
        413
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("       FOOD ANALYSIS SYSTEM")
    print("========================================")

    print()
    print("Base Directory:")
    print(BASE_DIR)

    print()
    print("Frontend:")
    print(FRONTEND_DIR)

    print()
    print("Uploads:")
    print(UPLOAD_FOLDER)

    print()
    print("History:")
    print("http://127.0.0.1:5000/history")

    print()
    print("Home:")
    print("http://127.0.0.1:5000")

    print()
    print("========================================")
    print()

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )