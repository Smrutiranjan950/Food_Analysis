import os

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

from preprocessing import preprocess_image

from model import predict_image
from model import get_food_info

from database import (
    create_table,
    save_analysis,
    get_history,
    delete_history
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


# ============================================================
# CREATE UPLOAD FOLDER
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_table()


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR
)


app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


app.config[
    "MAX_CONTENT_LENGTH"
] = 16 * 1024 * 1024


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE FOOD
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    if "food_image" not in request.files:

        return redirect(
            url_for("home")
        )


    file = request.files[
        "food_image"
    ]


    if file.filename == "":

        return redirect(
            url_for("home")
        )


    if not allowed_file(
        file.filename
    ):

        return (
            "Invalid image format. "
            "Please upload JPG, JPEG or PNG."
        )


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    filename = file.filename

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    file.save(
        image_path
    )


    try:

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        preprocessing_result = (
            preprocess_image(
                image_path
            )
        )


        features = (
            preprocessing_result[
                "features"
            ]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = predict_image(
            image_path
        )


        if not prediction[
            "success"
        ]:

            return (
                "Prediction failed: "
                +
                prediction.get(
                    "error",
                    "Unknown error"
                )
            )


        food_name = prediction[
            "food_name"
        ]


        confidence = prediction[
            "confidence"
        ]


        # ----------------------------------------------------
        # Nutrition information
        # ----------------------------------------------------

        food_info = get_food_info(
            food_name
        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        save_analysis(
            filename=filename,
            food_name=food_name,
            confidence=confidence,
            food_info=food_info
        )


        # ----------------------------------------------------
        # Show result
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

        return (
            "Error while analyzing image: "
            + str(e)
        )


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history"
)
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
# SERVE UPLOADED IMAGE
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# ERROR: FILE TOO LARGE
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return (
        "File is too large. "
        "Maximum allowed size is 16 MB.",
        413
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "       FOOD ANALYSIS SYSTEM"
    )

    print(
        "========================================"
    )

    print(
        "\nFrontend:"
    )

    print(
        FRONTEND_DIR
    )

    print(
        "\nUploads:"
    )

    print(
        UPLOAD_FOLDER
    )

    print(
        "\nHistory:"
    )

    print(
        "http://127.0.0.1:5000/history"
    )

    print(
        "\nHome:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\n========================================\n"
    )

    app.run(
        debug=True
    )