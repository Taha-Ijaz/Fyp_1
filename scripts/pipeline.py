# scripts/pipeline.py
import os
import uuid
import json
from datetime import datetime

# import your helper modules (they must be in scripts/ or in PYTHONPATH)
from pdf_extractor import extract_text_from_pdf
from preprocess import preprocess_text
from skill_extractor import extract_skills
# use semantic similarity instead of TF-IDF
from semantic_similarity import get_semantic_similarity as get_similarity


# ---- Configuration ----
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---- Simple template-based cover letter generator ----
def generate_cover_letter_simple(resume_text, jd_text, resume_skills, jd_skills, missing_skills):
    """
    Very simple, template-based cover letter.
    You can replace with an LLM later.
    """
    # Take short snippets for personalization
    resume_snippet = (resume_text[:500] + "...") if len(resume_text) > 500 else resume_text
    jd_snippet = (jd_text[:300] + "...") if len(jd_text) > 300 else jd_text

    matched_skills = [s for s in jd_skills if s in resume_skills]
    matched_text = ", ".join(matched_skills) if matched_skills else "relevant skills"

    missing_text = ", ".join(missing_skills) if missing_skills else "none"

    letter = f"""Dear Hiring Manager,

I am writing to apply for the position you described. Based on the job description provided, you are looking for: {', '.join(jd_skills)}.

My background includes experience with {matched_text}. A short summary of my experience: {resume_snippet}

I noticed the job mentions {', '.join(jd_skills)}. Skills I currently do not list on my resume are: {missing_text}. I am eager to learn and can quickly get up to speed on these.

I am excited about the opportunity and believe my experience would be a good match for this role.

Sincerely,
[Applicant Name]
"""
    return letter

# ---- Main pipeline function ----
def process_application(resume_pdf_path: str, job_description_text: str, save_outputs: bool = True):
    """
    Process a resume PDF and a pasted job description text.
    Returns a dictionary with:
      - cleaned_resume_text
      - cleaned_jd_text
      - resume_skills
      - jd_skills
      - missing_skills
      - similarity (float)
      - cover_letter (string)
      - saved_paths (optional)
    """
    # 1) Extract resume raw text
    raw_resume = extract_text_from_pdf(resume_pdf_path) or ""
    cleaned_resume = preprocess_text(raw_resume)

    # 2) Preprocess JD text
    cleaned_jd = preprocess_text(job_description_text or "")

    # 3) Extract skills (both)
    resume_skills = extract_skills(cleaned_resume)
    jd_skills = extract_skills(cleaned_jd)

    # normalize skills to lower-case strings for comparison
    resume_skills_norm = [s.lower() for s in resume_skills]
    jd_skills_norm = [s.lower() for s in jd_skills]

    # 4) Compute similarity
    try:
        similarity = float(get_similarity(cleaned_resume, cleaned_jd))
    except Exception:
        # fallback: similarity 0.0 if TF-IDF fails
        similarity = 0.0

    # 5) Missing skills (jd \ resume)
    missing_skills = [s for s in jd_skills_norm if s not in resume_skills_norm]
    # job recommendation based on similarity and missing skills
    recommendation = get_recommendation(similarity, missing_skills)


    # 6) Generate a simple cover letter
    cover_letter = generate_cover_letter_simple(cleaned_resume, cleaned_jd, resume_skills_norm, jd_skills_norm, missing_skills)

    # 7) Save outputs (optional)
    saved_paths = {}
    if save_outputs:
        uid = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resume_out = os.path.join(PROCESSED_DIR, f"resume_{uid}_{timestamp}.txt")
        jd_out = os.path.join(PROCESSED_DIR, f"jd_{uid}_{timestamp}.txt")
        letter_out = os.path.join(PROCESSED_DIR, f"cover_letter_{uid}_{timestamp}.txt")
        meta_out = os.path.join(PROCESSED_DIR, f"meta_{uid}_{timestamp}.json")

        with open(resume_out, "w", encoding="utf-8") as f:
            f.write(cleaned_resume)
        with open(jd_out, "w", encoding="utf-8") as f:
            f.write(cleaned_jd)
        with open(letter_out, "w", encoding="utf-8") as f:
            f.write(cover_letter)

        meta = {
            "uid": uid,
            "timestamp": timestamp,
            "resume_path": resume_pdf_path,
            "resume_saved": resume_out,
            "jd_saved": jd_out,
            "cover_letter_saved": letter_out,
            "similarity": similarity,
            "resume_skills": resume_skills_norm,
            "jd_skills": jd_skills_norm,
            "missing_skills": missing_skills
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        saved_paths = {
            "resume_text": resume_out,
            "jd_text": jd_out,
            "cover_letter": letter_out,
            "meta": meta_out
        }

    result = {
        "cleaned_resume": cleaned_resume,
        "cleaned_jd": cleaned_jd,
        "resume_skills": resume_skills_norm,
        "jd_skills": jd_skills_norm,
        "missing_skills": missing_skills,
        "similarity": similarity,
        "cover_letter": cover_letter,
        "saved_paths": saved_paths,
        "recommendation": recommendation

    }

    return result

# this is the function to generate recommendation based on similarity and missing skills
def get_recommendation(similarity: float, missing_skills: list):
    """
    Generate recommendation decision based on similarity score and missing skills.
    """
    # Strong match
    if similarity >= 0.70 and len(missing_skills) <= 2:
        return "STRONG MATCH – Recommended to Apply"

    # Medium match
    elif similarity >= 0.40:
        return "MODERATE MATCH – You Can Apply"

    # Weak match
    else:
        return "WEAK MATCH – Not Recommended"

if __name__ == "__main__":
    # small demonstration (replace path and text as needed)
    demo_resume = os.path.join("data", "raw", "resumes", "TAAHA-IJAZ-RESUME.pdf")
    demo_jd = """
We are hiring a Python Developer with strong experience in backend development,
REST APIs, and data processing. The candidate must be comfortable with frameworks
like FastAPI or Django. Experience with SQL databases, debugging, and Git is required.
Knowledge of Machine Learning is a plus.
"""

    output = process_application(demo_resume, demo_jd)
    print("Similarity:", output["similarity"])
    print("Resume skills:", output["resume_skills"])
    print("JD skills:", output["jd_skills"])
    print("Missing:", output["missing_skills"])
    print("Recommendation:", output["recommendation"])
    print("Cover letter saved to:", output["saved_paths"].get("cover_letter"))

