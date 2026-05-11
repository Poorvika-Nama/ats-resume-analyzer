# ATS Resume Analyzer & Skill Gap Report

An end-to-end ML system that compares a resume against a job description and outputs an ATS compatibility score, skill gap report, and personalised learning recommendations.

---

## What it does

- **ATS Score (0-100)** — predicts how well a resume matches a job description
- **Skill Gap Report** — classifies every required skill as Present / Partial / Missing  
- **Top Skills to Learn** — ranks missing skills by demand with direct tutorial links
- **Interactive Dashboard** — dark SaaS-style UI with animated score gauge and clickable skill badges

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Backend | Python, Flask |
| NLP | spaCy (en_core_web_sm), SBERT (all-MiniLM-L6-v2) |
| ML Model | XGBoost Regressor |
| Embeddings | Sentence Transformers |
| Deployment | Hugging Face Spaces |

---

## ATS Score Formula

ATS Score = (skill_overlap_ratio x 45) + (sbert_similarity x 25) + (exp_score x 20) + (edu_score x 10)

| Component | Weight | Description |
|---|---|---|
| Skill overlap | 45% | Exact skill match ratio |
| Semantic similarity | 25% | SBERT cosine similarity of full texts |
| Experience score | 20% | Actual vs required experience |
| Education score | 10% | Education level vs role requirement |

---

## Model Performance

| Metric | Value |
|---|---|
| MAE | 0.12 |
| RMSE | 0.20 |
| R2 | 0.9995 |

Trained on 24,840 resume-JD pairs using a Top-10 JDs per resume pairing strategy.

---

## Project Structure

ats-resume-analyzer/
- app.py                  Flask server and ML pipeline
- requirements.txt        Python dependencies
- ats_model.joblib        Trained XGBoost model
- feature_cols.json       Feature order for prediction
- templates/index.html    Dashboard UI
- notebooks/              Colab training notebooks
- assets/                 Model evaluation charts
- data/sample_input.json  Sample API input

---

## Run Locally

1. Clone the repo

git clone https://github.com/Poorvika-Nama/ats-resume-analyzer.git
cd ats-resume-analyzer

2. Install dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

3. Run the app

python app.py

4. Open in browser

http://localhost:5000

---

## API Usage

Endpoint: POST /analyze

Request body:
{
  "resume_text": "Your full resume text here",
  "jd_text": "Full job description text here",
  "jd_title": "Senior Data Scientist",
  "min_exp_years": 3
}

Response:
{
  "ats_score": 78.5,
  "ats_grade": "B",
  "present": ["python", "sql", "pandas"],
  "partial": [{"skill": "nlp", "matched_with": "spacy", "similarity": 0.81}],
  "missing": ["kubernetes", "spark"],
  "exp_years": 4.0,
  "edu_level": 3
}

---

## Datasets

| Dataset | Rows | Description |
|---|---|---|
| resumes_nlp.csv | 2,484 | Cleaned resumes with NLP-extracted skills |
| jd_nlp.csv | 1,068 | Job descriptions with required skills |
| feature_store.csv | 24,840 | Engineered features for all resume-JD pairs |

---

## Skill Gap Logic

Each required skill in the JD is classified as:

- Present — exact match found in resume skills
- Partial — SBERT cosine similarity >= 0.65 with a resume skill  
- Missing — no match above threshold

---

## Author

Your Name: Poorvika Nama ,
LinkedIn: https://www.linkedin.com/in/poorvika-nama-986185291/ ,
GitHub: https://github.com/Poorvika-Nama

---

## License

MIT License — free to use and modify.
