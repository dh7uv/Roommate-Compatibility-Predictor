# Roommate Compatibility Predictor

This project is now a local Flask website for matching students with compatible roommates using lifestyle preferences from a survey CSV.

## What it does

- Loads student responses from `responses.csv`
- Removes irrelevant identity and hostel allocation fields from clustering
- Standardizes meaningful lifestyle features
- Uses K-Means clustering to group similar students
- Uses cosine similarity to compute compatibility inside each cluster
- Filters matches to the same gender
- Shows the full compatibility list instead of only the top 3
- Provides descriptive explanations for each lifestyle field in the web interface

## Website modes

- `Existing student`: pick a student already in the CSV and view their full same-gender compatibility list
- `New profile input`: enter a lifestyle profile manually and view the full same-gender compatibility list

## Run the website

```bash
python app.py
```

Then open the local address shown in the terminal, usually:

`http://127.0.0.1:5000`

## Main files

- `app.py` runs the Flask website
- `roommate_predictor.py` contains preprocessing, clustering, compatibility scoring, and recommendation logic
- `templates/index.html` contains the web page structure
- `static/styles.css` contains the website styling

## Dataset path

The app first looks for:

`C:\Users\Dhruv Marfatia\Downloads\responses.csv`

If that file is not found, it falls back to:

`C:\Users\Dhruv Marfatia\Documents\New project\responses.csv`
