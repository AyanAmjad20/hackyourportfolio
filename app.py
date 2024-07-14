from flask import Flask, request, jsonify, render_template
import PyPDF2
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and TF-IDF Vectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))

# Define the function to clean resume text
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)  # Remove URLs
    clean_text = re.sub(r'\bRT\b|\bcc\b', ' ', clean_text)  # Remove "RT" and "cc"
    clean_text = re.sub(r'#\S+', '', clean_text)  # Remove hashtags
    clean_text = re.sub(r'@\S+', '  ', clean_text)  # Replace Twitter handles
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Remove punctuation
    clean_text = re.sub(r'[^\x00-\x7F]', ' ', clean_text)  # Remove non-ASCII characters
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Replace multiple spaces
    return clean_text

# Map category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        app.logger.error('No file part')
        return jsonify({'error': 'No file part'})

    file = request.files['resume']

    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith('.pdf'):
        try:
            resume_text = ''
            pdf = PyPDF2.PdfReader(file)  # Instantiate PdfReader directly
            for page in pdf.pages:
                resume_text += page.extract_text()
            
            if not resume_text:
                app.logger.error('No text found in the PDF.')
                return jsonify({'error': 'No text found in the PDF.'})
            
            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")

            app.logger.info(f'Prediction ID: {prediction_id}, Category: {category_name}')
            return jsonify({'category': category_name})
        except Exception as e:
            app.logger.error(f'Error processing PDF file: {e}')
            return jsonify({'error': f'Error processing PDF file: {e}'})
    else:
        app.logger.error('Invalid file type. Please upload a PDF.')
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'})

if __name__ == '__main__':
    app.run(debug=True)
