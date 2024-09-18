
# Flask Application for Resume Building, Interview Analysis, and Recommendation

This Flask application provides endpoints for resume building, interview processing, resume analysis, similarity scoring, and job recommendation. The application utilizes various models and libraries to achieve its functionality.

## Features

- **Resume Building**: Generate resumes based on user input.
- **Interview Processing**: Record and process interviews.
- **Resume Analysis**: Analyze resumes using provided job descriptions.
- **Similarity Scoring**: Calculate similarity scores between job descriptions and resumes.
- **Recommendation**: Provide recommendations based on job descriptions and resumes.

## Installation


### Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```


## Ollama Models

The following Ollama models are used:

- **Embedding Model**: `mxbai-embed-large`
- **Generation Model**: `wizardlm2`
- **Generation Model**: `llama3.1`
- **Generation Model**: `llama3.1:8b`

## API Endpoints

### 1. Generate a Resume

**URL**: `/resume-build`  
**Method**: `POST`  
**Request Body**:
```json
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "contact": "+1234567890",
  "education": "B.Sc. Computer Science",
  "experience": "5 years in software development",
  "skills": "Python, Flask, Django",
  "user_input": "I am interested in a software development role."
}
```
**Response**:
```json
{
  "resume": "Generated resume content here"
}
```

### 2. Display the Interview Page

**URL**: `/interview`  
**Method**: `GET`

### 3. Process the Recorded Interview

**URL**: `/process_interview`  
**Method**: `POST`  
**Response**:
```json
{
  "result": "Interview analysis result here"
}
```

### 4. Analyze a Resume

**URL**: `/analyse-resume`  
**Method**: `POST`  
**Request Body**:
```json
{
  "file_url": "https://utfs.io/f/uYCJGxAcJId8uk6is3AcJId8F3SNyhRlvDtHqWgfLOrVjk70"
}
```
**Response**:
```json
{
  "analysis": "Detailed analysis result here"
}
```

### 5. Calculate Similarity Score

**URL**: `/similarity-score`  
**Method**: `POST`  
**Request Body**:
```json
{
  "job_description": "Job description text here",
  "resume_url": "https://utfs.io/f/uYCJGxAcJId8uk6is3AcJId8F3SNyhRlvDtHqWgfLOrVjk70"
}
```
**Response**:
```json
{
  "score": 85,
  "details": "Similarity details here"
}
```

### 6. Get Recommendations

**URL**: `/recommendation`  
**Method**: `POST`  
**Request Body**:
```json
{
  "job_description": "Job description text here",
  "resume_url": "https://utfs.io/f/uYCJGxAcJId8uk6is3AcJId8F3SNyhRlvDtHqWgfLOrVjk70"
}
```
**Response**:
```json
{
  "recommendations": "Recommendations result here"
}
```

## Running the Application

To start the Flask server, run:

```bash
python main.py
```


