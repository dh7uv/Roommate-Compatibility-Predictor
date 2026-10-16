# Roommate Compatibility Predictor 🏠

A Python web application that recommends potential roommates by comparing lifestyle preferences and grouping similar profiles with a custom K-Means clustering implementation.

> **Project status:** Academic prototype. Compatibility scores are similarity estimates based on the available survey responses; they are not a guarantee that two people will live well together.

## Contents

- [Problem statement](#problem-statement)
- [Features](#features)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Setup and run](#setup-and-run)
- [Dataset format](#dataset-format)
- [Privacy and responsible use](#privacy-and-responsible-use)
- [Limitations and future work](#limitations-and-future-work)
- [Contributing](#contributing)

## Problem statement

Roommate conflicts can arise from differences in sleep schedules, cleanliness, noise tolerance, social activity, sharing habits, study routines, lighting, and temperature preferences. This project explores whether lifestyle survey data can help students discover potential roommates whose day-to-day habits are more compatible.

## Features

- Select an existing student profile from the CSV dataset.
- Enter a new profile using eight lifestyle preference fields.
- Preprocess survey responses and handle missing numeric answers.
- Scale lifestyle features and apply feature weights.
- Group profiles using a custom K-Means implementation.
- Recommend up to three same-cluster candidates and up to three candidates from other clusters, subject to the application's same-gender matching filter.
- Show a compatibility score and a text explanation for recommended candidates.
- Provide a Flask-based browser interface.

## How it works

1. **Load data:** the Flask application reads `responses.csv` with pandas.
2. **Prepare features:** the prediction module maps survey column names to eight lifestyle features and excludes contact, room-location, and other non-lifestyle columns from the distance calculation.
3. **Handle missing values:** numeric feature values are coerced to numbers and missing values are filled using the feature median.
4. **Scale and weight:** features are standardized and multiplied by configured feature weights.
5. **Cluster:** the predictor selects a cluster count and applies its custom K-Means implementation.
6. **Recommend:** candidates are ranked using the predictor's distance/compatibility logic, with separate results for the same cluster and other clusters.
7. **Display:** Flask renders the selected profile's recommendations in the HTML template.

The current feature weights are configured in `roommate_predictor.py`. They are design choices, not weights learned from a validated outcome dataset. Review the code before making claims about predictive accuracy.

## Technology stack

- **Python** — application and matching logic
- **Flask** — web server and routing
- **Pandas** — CSV loading and data preparation
- **NumPy** — numeric operations used by the predictor
- **HTML/CSS/JavaScript** — browser interface in `templates/` and `static/`
- **CSV** — current data storage format

## Repository structure

```text
Roommate-Compatibility-Predictor/
├── app.py                  # Flask routes, form handling, dataset updates
├── roommate_predictor.py   # Preprocessing, clustering and recommendations
├── responses.csv           # Survey dataset (keep private if it contains real data)
├── templates/              # HTML templates
├── static/                 # CSS, JavaScript and other static assets
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and setup guide
├── LICENSE                 # Project license (confirm before publishing)
├── CONTRIBUTING.md         # Contribution workflow
├── SECURITY.md             # Security and privacy reporting guidance
└── docs/
    ├── MODEL_METHODOLOGY.md
    └── TESTING_CHECKLIST.md
```

## Setup and run

### Requirements

- Python 3.10 or newer is recommended.
- Git (optional, for cloning the repository).
- A local copy of `responses.csv` with the expected survey columns.

### Windows PowerShell

```powershell
git clone https://github.com/dh7uv/Roommate-Compatibility-Predictor.git
cd Roommate-Compatibility-Predictor

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

### macOS / Linux

```bash
git clone https://github.com/dh7uv/Roommate-Compatibility-Predictor.git
cd Roommate-Compatibility-Predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

> Run this academic prototype locally. Before deploying it publicly, disable Flask debug mode, move real student data out of the repository, add appropriate access controls, and review the security/privacy notes below.

## Dataset format

The predictor expects a `responses.csv` file containing a `Full name` column and at least three usable lifestyle features. The intended features are:

| Feature | Typical source question |
|---|---|
| Sleep schedule | What is your usual sleep time? |
| Study habits | How many hours do you study daily? |
| Lights preference | Do you prefer lights on while sleeping? |
| Cleanliness | How clean do you keep your room? |
| Noise tolerance | How much noise can you tolerate while studying? |
| Social activity | How often do you invite friends to your room? |
| Sharing comfort | How comfortable are you in sharing your belongings? |
| Temperature preference | How warm/cold do you like your room? |

The code also recognizes several normalized feature aliases. Use the exact survey headers from your existing CSV unless you have updated the alias mapping in `roommate_predictor.py` as well.

**Do not upload real student names, email addresses, room numbers, hostel locations, or identifiable feedback to a public repository.** Replace the tracked dataset with a synthetic/demo dataset or keep the real dataset outside Git and document how to provide it locally.

## Privacy and responsible use

- Collect only information needed for matching and obtain informed consent.
- Do not commit real survey responses, emails, room numbers, or sensitive free-text feedback.
- If the repository has already exposed real data, deleting the file in a new commit is not enough: remove it from Git history where appropriate and notify affected data owners.
- Avoid presenting a compatibility score as an objective judgement of a person.
- Explain that the system is a decision-support prototype and let students make their own choices.
- The current implementation writes manually entered profiles back to the CSV. Review this behaviour before using shared or production data.
- The current Flask entry point uses debug mode when run directly; debug mode must not be enabled on a public deployment.

## Limitations and future work

- Validate recommendations with consented feedback and appropriate evaluation metrics.
- Compare the current clustering method with baselines such as nearest-neighbour ranking without clustering.
- Review whether the feature weights reflect student priorities; support configurable weights or user-ranked preferences.
- Improve validation for names, duplicate profiles, numeric ranges, missing columns, and malformed survey data.
- Add automated unit tests and continuous integration.
- Separate private data from source code and replace CSV writes with a properly designed persistence layer if multi-user deployment is planned.
- Add screenshots of the actual running interface after testing it locally.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please do not submit real or identifiable student data in issues or pull requests.

## License

See [LICENSE](LICENSE). Confirm that you have permission to publish all code, survey data, screenshots, and other assets before choosing a public license.
