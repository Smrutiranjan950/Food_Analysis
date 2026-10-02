import sqlite3
import os
from datetime import datetime


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "food_analysis.db"
)


# ============================================================
# CONNECT TO DATABASE
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CREATE TABLE
# ============================================================

def create_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT NOT NULL,

            food_name TEXT NOT NULL,

            confidence REAL,

            calories REAL,

            protein REAL,

            carbs REAL,

            fat REAL,

            fiber REAL,

            analyzed_at TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(
    filename,
    food_name,
    confidence,
    food_info
):

    calories = None
    protein = None
    carbs = None
    fat = None
    fiber = None

    if food_info:

        calories = food_info.get(
            "calories_kcal_100g"
        )

        protein = food_info.get(
            "protein_g_100g"
        )

        carbs = food_info.get(
            "carbs_g_100g"
        )

        fat = food_info.get(
            "fat_g_100g"
        )

        fiber = food_info.get(
            "fiber_g_100g"
        )

    analyzed_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO analysis_history
        (
            filename,
            food_name,
            confidence,
            calories,
            protein,
            carbs,
            fat,
            fiber,
            analyzed_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            food_name,
            confidence,
            calories,
            protein,
            carbs,
            fat,
            fiber,
            analyzed_at
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# GET ALL HISTORY
# ============================================================

def get_history():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM analysis_history
        ORDER BY id DESC
        """
    )

    history = cursor.fetchall()

    connection.close()

    return history


# ============================================================
# DELETE HISTORY
# ============================================================

def delete_history():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM analysis_history
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_table()


if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "       FOOD ANALYSIS DATABASE"
    )

    print(
        "========================================"
    )

    print(
        "Database:"
    )

    print(
        DATABASE_PATH
    )

    print(
        "\nDatabase initialized successfully."
    )

    print(
        "========================================"
    )