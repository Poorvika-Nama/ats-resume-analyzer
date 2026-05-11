
import os, re, json, numpy as np, joblib
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

app = Flask(__name__)

# ── Load models ────────────────────────────────────────────────────────
nlp   = spacy.load("en_core_web_sm")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
ats_model = joblib.load("ats_model.joblib")
with open("feature_cols.json") as f:
    feature_cols = json.load(f)

# ── Skill dictionary ───────────────────────────────────────────────────
SKILL_DICT = [
    "python","java","javascript","typescript","c#","c++","go","rust",
    "kotlin","swift","r","scala","php","ruby","html","css","react",
    "angular","vue","node.js","django","flask","fastapi","spring",
    "asp.net",".net","mysql","postgresql","mongodb","redis","cassandra",
    "oracle","sql server","dynamodb","elasticsearch","aws","azure","gcp",
    "docker","kubernetes","terraform","ansible","jenkins","git","ci/cd",
    "linux","machine learning","deep learning","nlp","tensorflow","pytorch",
    "scikit-learn","pandas","numpy","spark","airflow","agile","scrum",
    "jira","leadership","communication","project management","excel",
    "power bi","tableau","sql","data analysis","computer vision",
    "reinforcement learning","hadoop","kafka","azure devops","figma",
    "adobe xd","photoshop","illustrator","seo","google analytics",
    "rest api","graphql","microservices","devops","mlops",
    "data visualization","statistics","probability"
]

ALIAS_MAP = {
    "ml":"machine learning","ai":"machine learning",
    "js":"javascript","ts":"typescript",
    "k8s":"kubernetes","node":"node.js",
    "mongo":"mongodb","postgres":"postgresql",
    "amazon web services":"aws","google cloud":"gcp",
    "microsoft azure":"azure","deep neural network":"deep learning",
    "natural language processing":"nlp","cv":"computer vision",
    "react.js":"react","reactjs":"react","vue.js":"vue",
    "angularjs":"angular","sklearn":"scikit-learn",
    "scikit learn":"scikit-learn","tf":"tensorflow",
    "spacy":"nlp","bert":"nlp","llm":"nlp",
    "team management":"leadership","team leadership":"leadership",
    "people management":"leadership","verbal communication":"communication"
}

ROLE_EDU_REQ = {
    "fresher":3,"junior":3,"mid":3,
    "unknown":3,"senior":4,"lead":4,"manager":4
}

# ── Helper functions ───────────────────────────────────────────────────
def extract_skills(text):
    text_lower = text.lower()
    for alias, canonical in ALIAS_MAP.items():
        text_lower = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, text_lower)
    found = set()
    for skill in SKILL_DICT:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)
    doc = nlp(text_lower)
    for ent in doc.ents:
        val = ent.text.strip().lower()
        if val in SKILL_DICT:
            found.add(val)
    return list(found)

def extract_experience(text):
    patterns = [
        r"(\d+)\+?\s*years? of experience",
        r"experience of (\d+)\+?\s*years?",
        r"(\d+)\+?\s*years? experience",
        r"(\d+)\+?\s*yrs?"
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return float(m.group(1))
    return 0.0

def extract_edu_level(text):
    t = text.lower()
    if any(w in t for w in ["phd","ph.d","doctorate"]):           return 5
    if any(w in t for w in ["master","msc","m.sc","mba","m.tech"]):return 4
    if any(w in t for w in ["bachelor","bsc","b.sc","b.tech","be ","b.e"]):return 3
    if any(w in t for w in ["associate","diploma"]):               return 2
    if any(w in t for w in ["high school","12th","hsc"]):          return 1
    return 0

def infer_role_level(title):
    t = title.lower()
    if any(w in t for w in ["intern","fresher","trainee","graduate"]): return "fresher"
    if any(w in t for w in ["junior","jr.","entry"]):                  return "junior"
    if any(w in t for w in ["senior","sr.","lead","principal","staff"]):return "senior"
    if any(w in t for w in ["manager","director","head","vp","chief"]): return "manager"
    return "mid"

def skill_gap_report(resume_skills, jd_skills, threshold=0.65):
    present, partial, missing = [], [], []
    resume_set = set(resume_skills)
    for skill in jd_skills:
        if skill in resume_set:
            present.append(skill)
        else:
            jd_emb   = sbert.encode([skill])
            best_score, best_match = 0.0, None
            for rs in resume_skills:
                score = cosine_similarity(jd_emb, sbert.encode([rs]))[0][0]
                if score > best_score:
                    best_score, best_match = score, rs
            if best_score >= threshold:
                partial.append({
                    "skill": skill,
                    "matched_with": best_match,
                    "similarity": round(float(best_score), 3)
                })
            else:
                missing.append(skill)
    return present, partial, missing

def run_analysis(resume_text, jd_text, jd_title, min_exp_years):
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    overlap       = set(resume_skills) & set(jd_skills)
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / len(jd_skills) if jd_skills else 0.0

    exp_years  = extract_experience(resume_text)
    edu_level  = extract_edu_level(resume_text)
    role_level = infer_role_level(jd_title)
    edu_req    = ROLE_EDU_REQ.get(role_level, 3)

    exp_score = min(exp_years / min_exp_years, 1.0) if min_exp_years > 0 else 1.0
    edu_score = min(edu_level / edu_req, 1.0)       if edu_level  > 0 else 0.5

    res_emb   = sbert.encode([resume_text])
    jd_emb    = sbert.encode([jd_text])
    sbert_sim = float(cosine_similarity(res_emb, jd_emb)[0][0])

    features = {
        "skill_overlap_count":   overlap_count,
        "skill_overlap_ratio":   overlap_ratio,
        "resume_total_skills":   len(resume_skills),
        "experience_years":      exp_years,
        "min_exp_years":         min_exp_years,
        "exp_meets_requirement": 1 if exp_years >= min_exp_years else 0,
        "education_level":       edu_level,
        "sbert_similarity":      sbert_sim,
        "exp_score":             exp_score,
        "edu_score":             edu_score
    }

    X         = np.array([[features[c] for c in feature_cols]])
    ats_score = float(ats_model.predict(X)[0])
    ats_score = round(min(max(ats_score, 0), 100), 2)

    if   ats_score >= 75: grade = "A"
    elif ats_score >= 60: grade = "B"
    elif ats_score >= 45: grade = "C"
    elif ats_score >= 30: grade = "D"
    else:                 grade = "F"

    present, partial, missing = skill_gap_report(resume_skills, jd_skills)

    return {
        "ats_score":      ats_score,
        "ats_grade":      grade,
        "present":        present,
        "partial":        partial,
        "missing":        missing,
        "resume_skills":  resume_skills,
        "jd_skills":      jd_skills,
        "features":       features,
        "exp_years":      exp_years,
        "edu_level":      edu_level
    }

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data          = request.get_json()
    resume_text   = data.get("resume_text", "")
    jd_text       = data.get("jd_text", "")
    jd_title      = data.get("jd_title", "")
    min_exp_years = float(data.get("min_exp_years", 0))

    if not resume_text or not jd_text or not jd_title:
        return jsonify({"error": "resume_text, jd_text and jd_title are required"}), 400

    result = run_analysis(resume_text, jd_text, jd_title, min_exp_years)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
