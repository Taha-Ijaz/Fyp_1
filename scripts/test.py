from skill_extractor import extract_skills
from similarity import get_similarity
from pdf_extractor import extract_text_from_pdf
from preprocess import preprocess_text

pdf_path = "data/raw/resumes/TAAHA-IJAZ-RESUME.pdf"    # only one test file
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = preprocess_text(raw_text)

print("RAW:\n", raw_text[:500])
print("\nCLEANED:\n", cleaned_text[:500])
output_path = "data/processed/resume1_cleaned.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Saved cleaned text to:", output_path)
# ---------------------------------------------
# STEP 2 — PROCESS JOB DESCRIPTION (PASTED TEXT)
# ---------------------------------------------
job_description = """
We are hiring a Python developer with experience in machine learning,
NLP, and REST API development. The candidate must understand data
preprocessing and model building.
"""
clean_jd = preprocess_text(job_description)

jd_output = "data/processed/jd1_cleaned.txt"
with open(jd_output, "w", encoding="utf-8") as f:
    f.write(clean_jd)

print("\nCleaned JD saved to:", jd_output)
print("\nCLEANED JD:\n", clean_jd[:300])

# ---------------------------------------------
score = get_similarity(cleaned_text, clean_jd)

print("\n===================================")
print("RESUME–JOB SIMILARITY SCORE:", score)
print("===================================")
resume_skills = extract_skills(cleaned_text)
print("Skills found in resume:", resume_skills)
jd_skills = extract_skills(clean_jd)
print("Skills found in JD:", jd_skills)
print("===================================")
missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
print("Missing skills:", missing_skills)

