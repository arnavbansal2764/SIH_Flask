from flask import Flask, request, jsonify, render_template
from interview_main import process_interview, record_interview
from resume_build import load_user_data, generate_resume
from analyse_resume import load_data, get_bot_response
from similarity_score import download_file, pdf_to_text, calculate_similarity_score
from recommendation import analyze_resume
import os
app = Flask(__name__)

# Route for generating a resume
@app.route('/resume-build', methods=['POST'])
def resume_build_route():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    contact = data.get("contact")
    education = data.get("education")
    experience = data.get("experience")
    skills = data.get("skills")
    user_input = data.get("user_input")

    if not all([name, email, contact, education, experience, skills, user_input]):
        return jsonify({"error": "Missing fields"}), 400

    user_data = f"""
    Name: {name}
    Email: {email}
    Phone: {contact}
    Education: {education}
    Experience: {experience}
    Skills: {skills}
    """
    
    load_user_data(user_data, "1")  # Load data for user with ID 1
    resume = generate_resume(user_input, "1")  # Generate resume for user with ID 1
    
    return jsonify({"resume": resume})


# Route to display the interview page
@app.route('/interview')
def interview_route():
    return render_template('interview.html')


# Route to process the recorded interview
@app.route('/process_interview', methods=['POST'])
def process_interview_route():
    # Process the interview and generate analysis
    result = process_interview()
    return jsonify(result)

@app.route('/analyse-resume', methods=['POST'])
def analyse_resume():
    data = request.json
    file_url = data.get("file_url")

    if not file_url:
        return jsonify({"error": "Missing file URL"}), 400

    try:
        # Download and process the resume
        load_data(file_url, "1")

        job_description = """
        Objectives of this role:
        Collaborate with product design and engineering teams to develop an understanding of needs
        Research and devise innovative statistical models for data analysis
        Communicate findings to all stakeholders
        Enable smarter business processes by using analytics for meaningful insights
        Keep current with technical and industry developments

        Responsibilities:
        Serve as lead data strategist to identify and integrate new datasets that can be leveraged through our product capabilities, and work closely with the engineering team in the development of data products
        Execute analytical experiments to help solve problems across various domains and industries
        Identify relevant data sources and sets to mine for client business needs, and collect large structured and unstructured datasets and variables
        Devise and utilize algorithms and models to mine big-data stores; perform data and error analysis to improve models; clean and validate data for uniformity and accuracy
        Analyze data for trends and patterns, and interpret data with clear objectives in mind
        Implement analytical models in production by collaborating with software developers and machine-learning engineers

        Required skills and qualifications:
        Seven or more years of experience in data science
        Proficiency with data mining, mathematics, and statistical analysis
        Advanced experience in pattern recognition and predictive modeling
        Experience with Excel, PowerPoint, Tableau, SQL, and programming languages (ex: Java/Python, SAS)
        Ability to work effectively in a dynamic, research-oriented group that has several concurrent projects

        Preferred skills and qualifications:
        Bachelor’s degree (or equivalent) in statistics, applied mathematics, or related discipline
        Two or more years of project management experience
        Professional certification
        """

        # Generate a detailed analysis by passing the job description
        result = get_bot_response(job_description)
        return jsonify({"analysis": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/similarity-score', methods=['POST'])
def similarity_score():
    try:
        data = request.json
        job_description = data.get('job_description')
        resume_url = data.get('resume_url')

        if not job_description or not resume_url:
            return jsonify({"error": "job_description and resume_url are required"}), 400

        # Step 1: Download the resume file
        resume_file = download_file(resume_url)

        # Step 2: Convert the downloaded PDF to text
        resume_text = pdf_to_text(resume_file)

        # Step 3: Calculate the similarity score
        scores = calculate_similarity_score(job_description, resume_text)

        # Step 4: Cleanup the downloaded file
        if os.path.exists(resume_file):
            os.remove(resume_file)

        return jsonify(scores)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommendation', methods=['POST'])
def recommendation():
    data = request.get_json()
    job_description = data.get('job_description')
    resume_url = data.get('resume_url')

    if not job_description or not resume_url:
        return jsonify({"error": "Missing job description or resume URL"}), 400

    recommendation_response = analyze_resume(job_description, resume_url)

    if "error" in recommendation_response:
        return jsonify(recommendation_response), 400

    return jsonify(recommendation_response), 200


if __name__ == "__main__":
    app.run(debug=True)
