from preprocess import preprocess_resume, preprocess_job_description
from similarity import get_similarity

# Sample text (You can replace with real texts)
resume_text = """
I am a Python developer with experience in machine learning and NLP.
"""

job_description_text = """
We are looking for a Python developer skilled in NLP and machine learning.
"""

# Preprocess both texts
resume_clean = preprocess_resume(resume_text)
jd_clean = preprocess_job_description(job_description_text)

# Get similarity score
score = get_similarity(resume_clean, jd_clean)

print("Similarity Score:", score)
