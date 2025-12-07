import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Predefined list of technical skills (expand anytime)
TECH_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "html", "css",
    "django", "flask", "react", "node", "sql", "mysql", "postgresql",
    "nlp", "machine learning", "deep learning", "tensorflow",
    "pytorch", "pyspark", "aws", "azure", "docker", "rest api",
    "git", "linux", "computer vision", "api", "ml"
]

def extract_skills(text):
    """
    Extracts technical skills from JD or resume text using spaCy + keyword matching.
    """
    doc = nlp(text)
    found_skills = set()

    # 1. Check for each word in the text using spaCy tokenization
    for token in doc:
        token_text = token.text.lower()
        if token_text in TECH_SKILLS:
            found_skills.add(token_text)

    # 2. Check multi-word skills (like "machine learning", "deep learning")
    text_lower = text.lower()
    for skill in TECH_SKILLS:
        if skill in text_lower:
            found_skills.add(skill)

    return list(found_skills)
