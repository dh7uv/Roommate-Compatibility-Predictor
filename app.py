from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd

from roommate_predictor import DISPLAY_NAMES, RoommateCompatibilityPredictor


BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATASET = BASE_DIR / "responses.csv"

FIELD_HELP = {
    "sleep_time": {
        "label": "Sleep Schedule",
        "description": "Choose how late or early you usually sleep. Lower values mean earlier sleeping habits, while higher values mean later nights.",
    },
    "study_hours": {
        "label": "Study Habits",
        "description": "Estimate your daily study effort. Lower values mean fewer study hours, while higher values mean a stronger study-focused routine.",
    },
    "lights_preference": {
        "label": "Lights Preference",
        "description": "Choose how comfortable you are with lights staying on while sleeping. Lower values prefer darker rooms; higher values are more comfortable with lights on.",
    },
    "cleanliness_level": {
        "label": "Cleanliness",
        "description": "Describe how tidy and organized you keep your room. Higher values represent a cleaner and more structured living space.",
    },
    "noise_tolerance": {
        "label": "Noise Tolerance",
        "description": "Describe how much background noise you can tolerate while studying or resting. Higher values mean you are more comfortable with noise.",
    },
    "social_activity": {
        "label": "Social Activity",
        "description": "Choose how often you like to invite friends or socialize in your room. Higher values mean a more social room environment.",
    },
    "sharing_comfort": {
        "label": "Sharing Comfort",
        "description": "Describe how comfortable you are with sharing personal belongings and common items with a roommate.",
    },
    "temperature_preference": {
        "label": "Room Temperature Preference",
        "description": "Choose how warm or cold you generally like your room to feel. Matching this helps avoid daily comfort conflicts.",
    },
}


def resolve_dataset_path() -> Path:
    return LOCAL_DATASET


def save_manual_user(dataset_path: Path, name: str, gender: str, manual_values: dict[str, str]) -> None:
    df = pd.read_csv(dataset_path)

    expected_defaults = {
        "Timestamp": "",
        "Email address": "",
        "Full name": "",
        "Sex": "",
        "Outlook email ID": "",
        "Hostel Block": "",
        "Floor": "",
        "Room number": "",
        "Current Roommate Name": "None",
        "What is your usual sleep time? ": "",
        "Do you prefer lights on while sleeping?": "",
        "How clean do you keep your room?": "",
        "How much noise can you tolerate while studying?": "",
        "How often do you invite friends to your room?": "",
        "How comfortable are you in sharing your belongings": "",
        "How warm/cold do you like your room?": "",
        "Please provide an honest feedback about your current roommate": "",
        "What is their usual sleep time? ": "",
        "Do they prefer lights on while sleeping?": "",
        "How clean do they keep the room?": "",
        "How noisy are they?(1-5)": "",
        "How often do they invite friends to the room?": "",
        "How warm/cold do they like the room?": "",
        "How many hours do you study daily?": "",
    }

    for column, default in expected_defaults.items():
        if column not in df.columns:
            df[column] = default

    row_data = {
        "Full name": name,
        "Sex": gender,
        "Current Roommate Name": "None",
        "What is your usual sleep time? ": int(manual_values["sleep_time"]),
        "Do you prefer lights on while sleeping?": int(manual_values["lights_preference"]),
        "How clean do you keep your room?": int(manual_values["cleanliness_level"]),
        "How much noise can you tolerate while studying?": int(manual_values["noise_tolerance"]),
        "How often do you invite friends to your room?": int(manual_values["social_activity"]),
        "How comfortable are you in sharing your belongings": int(manual_values["sharing_comfort"]),
        "How warm/cold do you like your room?": int(manual_values["temperature_preference"]),
        "How many hours do you study daily?": int(manual_values["study_hours"]),
    }

    existing_rows = df.index[
        df["Full name"].fillna("").astype(str).str.strip().str.lower() == name.strip().lower()
    ].tolist()

    if existing_rows:
        for column, value in row_data.items():
            df.loc[existing_rows[0], column] = value
    else:
        new_row = {column: expected_defaults.get(column, "") for column in df.columns}
        new_row.update(row_data)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(dataset_path, index=False)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    dataset_path = resolve_dataset_path()
    predictor = RoommateCompatibilityPredictor()

    context = {
        "dataset_path": str(dataset_path),
        "mode": "existing",
        "manual_name": "",
        "users": [],
        "selected_user": "",
        "gender": "Male",
        "manual_values": {field: "3" for field in FIELD_HELP},
        "same_cluster_results": [],
        "outside_cluster_results": [],
        "status": "",
        "field_help": [
            {"name": field, **FIELD_HELP[field]}
            for field in [
                "sleep_time",
                "study_hours",
                "lights_preference",
                "cleanliness_level",
                "noise_tolerance",
                "social_activity",
                "sharing_comfort",
                "temperature_preference",
            ]
        ],
    }

    if not dataset_path.exists():
        context["status"] = f"Dataset not found at {dataset_path}. Please place responses.csv there."
        return render_template("index.html", **context)

    try:
        predictor.fit(dataset_path)
    except Exception as exc:
        context["status"] = f"Failed to load dataset: {exc}"
        return render_template("index.html", **context)

    users = predictor.available_users()
    context["users"] = users
    if users:
        context["selected_user"] = users[0]

    if request.method == "POST":
        mode = request.form.get("mode", "existing")
        context["mode"] = mode
        context["manual_name"] = request.form.get("manual_name", "").strip()
        context["selected_user"] = request.form.get("selected_user", context["selected_user"])
        context["gender"] = request.form.get("gender", "Male")

        manual_values = {}
        for field in FIELD_HELP:
            manual_values[field] = request.form.get(field, "3")
        context["manual_values"] = manual_values

        try:
            if mode == "existing":
                if not context["selected_user"]:
                    raise ValueError("Please choose a student from the dataset.")
                same_cluster_results = predictor.recommend_for_user(context["selected_user"], top_n=3)
                outside_cluster_results = predictor.top_outside_cluster_for_user(context["selected_user"], top_n=3)
                subject = context["selected_user"]
            else:
                if not context["manual_name"]:
                    raise ValueError("Please enter a name for the new user profile.")
                save_manual_user(dataset_path, context["manual_name"], context["gender"], manual_values)
                predictor.fit(dataset_path)
                profile = {
                    field: float(manual_values[field])
                    for field in predictor.feature_columns
                }
                profile["Full name"] = context["manual_name"]
                profile["Sex"] = context["gender"]
                same_cluster_results = predictor.recommend_for_profile(profile, top_n=3)
                outside_cluster_results = predictor.top_outside_cluster_for_profile(profile, top_n=3)
                context["users"] = predictor.available_users()
                subject = context["manual_name"]

            context["same_cluster_results"] = same_cluster_results
            context["outside_cluster_results"] = outside_cluster_results
            if same_cluster_results:
                context["status"] = (
                    f"Showing the top same-gender matches for {subject}, followed by top same-gender matches from other clusters."
                )
            else:
                context["status"] = (
                    f"No same-gender matches were found in the assigned cluster for {subject}."
                )
        except Exception as exc:
            context["status"] = f"Prediction failed: {exc}"
    else:
        feature_names = ", ".join(
            DISPLAY_NAMES.get(feature, feature) for feature in predictor.feature_columns
        )
        context["status"] = (
            f"Loaded {len(predictor.model_df) if predictor.model_df is not None else 0} student profiles. "
            f"Current matching features: {feature_names}."
        )

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
