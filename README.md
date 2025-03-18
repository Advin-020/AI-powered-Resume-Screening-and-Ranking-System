# AI-powered-Resume-Screening-and-Ranking-System
This project is an AI-driven Resume Screening and Ranking System designed to automate and optimize the recruitment process. It leverages Natural Language Processing (NLP) techniques to compare resumes against job descriptions and rank candidates based on relevance.
# How It Works
To achieve this, the system utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to transform textual data into numerical form. 
It then applies cosine similarity to measure the 'degree of relevance' between a job description and each resume , which return a numeric value. 
The resumes are ranked based on their similarity scores.
The user interface is built with Streamlit, which provides a simple and user-friendly UI and handles both input and output.
