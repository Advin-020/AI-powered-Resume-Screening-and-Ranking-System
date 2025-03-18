import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Parsing - extracting text from PDFs using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or "" 
    return text

# Ranking resumes - using cosine similarity
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit UI
st.title("📝 AI Resume Screening & Candidate Ranking System")

# Input job description
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Input resume PDFs
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Ranking resumes based on cosine similarity score
if uploaded_files and job_description:
    st.header("🏆 Ranking Resumes")
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    # Input N - To display top N resumes
    top_n = st.number_input("Select top N resumes", min_value=1, max_value=len(uploaded_files), value=min(5, len(uploaded_files)), step=1)
    results = results.head(top_n)
    results.index = range(1, len(results) + 1)
    st.write(results)
