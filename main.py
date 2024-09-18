from flask import Flask, request, jsonify, render_template
from resume_build import load_user_data, generate_resume
from similarity_score import download_file, pdf_to_text, calculate_similarity_score
from recommendation import analyze_resume
import os
import wave
import pyaudio
import asyncio
from pydub import AudioSegment
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig
import speech_recognition as sr
import ollama
import os
import requests
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader

app = Flask(__name__)
# Initialize ChromaDB client and collection
client = chromadb.Client()
try:
    collection = client.create_collection(name="docs")
except chromadb.db.base.UniqueConstraintError:
    collection = client.get_collection(name="docs")


# Function to download PDF
def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("PDF downloaded successfully.")

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = os.path.join(os.getcwd(), "output.wav")  # Save in the current directory

# Initialize lists for storing results
new_list = []
emotions = []
text_segments = []

# Function to record audio
def record_audio():
    # Create PyAudio object
    p = pyaudio.PyAudio()

    # Open a new stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    # Record audio
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    # Stop the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write the audio data to a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Speech-to-text function for full audio
def stt_full():
    recognizer = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        text_segments.append(f"Complete answer: {text}")
    except sr.UnknownValueError:
        print("Google Web Speech could not understand the audio in full answer")
    except sr.RequestError:
        print("Could not request results from Google Web Speech API for the full answer")

# Process individual segments using Hume and Google Speech-to-Text
async def process_segment(segment, segment_index):
    segment_filename = f"output_segment_{segment_index}.wav"
    segment.export(segment_filename, format="wav")

    client = HumeStreamClient("CJffluuY10Z47dNMZSMs4WQ7eBparPq0XYWJduyczGMk9OQO")
    config = ProsodyConfig()

    async with client.connect([config]) as socket:
        result = await socket.send_file(segment_filename)
        result = result['prosody']['predictions']
        result = result[0]['emotions']

    top_3_emotions = sorted(result, key=lambda x: x['score'], reverse=True)[:3]
    new_list.append(top_3_emotions)

    current_emotions = [f"{emotion['name']} : {emotion['score']}" for emotion in top_3_emotions]
    emotions.append(current_emotions)

    recognizer = sr.Recognizer()
    with sr.AudioFile(segment_filename) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        text_segments.append(f"Text for segment {segment_index}: {text}")
    except sr.UnknownValueError:
        print(f"Google Web Speech could not understand the audio in segment {segment_index}")
    except sr.RequestError:
        print(f"Could not request results from Google Web Speech API for segment {segment_index}")

# Function to generate summary based on emotions and text
def generate_summary(emotions, text_segments, question):
    prompt = f"""
You have to judge the user's answer according to what they have spoken (text) and how they have spoken (emotions). The user does not know that the text has been divided into segments so just give a summary, give tips to the user about where and how they can improve.

question : {question}

{text_segments[-1]}

{text_segments[0]}
{text_segments[1]}
{text_segments[2]}
{text_segments[3]}
{text_segments[4]}
{text_segments[5]}

Top 3 emotions for segment 0:
{emotions[0][0]}
{emotions[0][1]}
{emotions[0][2]}

Top 3 emotions for segment 1:
{emotions[1][0]}
{emotions[1][1]}
{emotions[1][2]}

Top 3 emotions for segment 2:
{emotions[2][0]}
{emotions[2][1]}
{emotions[2][2]}

Top 3 emotions for segment 3:
{emotions[3][0]}
{emotions[3][1]}
{emotions[3][2]}

Top 3 emotions for segment 4:
{emotions[4][0]}
{emotions[4][1]}
{emotions[4][2]}

Top 3 emotions for segment 5:
{emotions[5][0]}
{emotions[5][1]}
{emotions[5][2]}
"""
    output = ollama.generate(model="llama3.1", prompt=prompt)
    return output["response"]

# Main function to process audio
async def measurer():
    audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
    segment_length = len(audio) // 6
    audio_segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(6)]

    tasks = [process_segment(segment, i) for i, segment in enumerate(audio_segments)]
    await asyncio.gather(*tasks)

    stt_full()
    return generate_summary(emotions, text_segments, "What is a linked list?")


# Function to load and process the PDF
def load_data(path, n):
    loader = PyMuPDFLoader(path)
    data = loader.load()

    list1 = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    text_splitter.split_documents(data)

    for i in range(len(data)):
        data_text = data[i].page_content
        data_text = data_text.replace("\n", "")
        list1.append(data_text)

    for i, data in enumerate(list1):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=data)
        embedding = response["embedding"]
        collection.add(ids=[str(i) + str(n)], embeddings=[embedding], documents=[data])

    print("Data Loaded")


# Function to get bot response
def get_bot_response(user_input):
    prompt = user_input
    response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    data = results["documents"][0][0]
    print("Data Given To Ollama ")
    output = ollama.generate(
        model="wizardlm2",
        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
    )
    return output["response"]

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

@app.route('/process_audio', methods=['POST'])
def process_audio():
    record_audio()  # Record audio from microphone
    result = asyncio.run(measurer())  # Run async tasks to process the audio
    return result  # Return the result to the webpage

# Define the Flask route for PDF processing and scoring
@app.route('/analyse-resume', methods=['GET'])
def calc_score():
    # Get the PDF URL from query params
    pdf_url = request.args.get('pdf_url')
    if not pdf_url:
        return jsonify({"error": "No PDF URL provided"}), 400

    # Define the path to save the downloaded PDF
    save_path = 'downloaded_resume.pdf'

    # Step 1: Download the PDF
    try:
        download_pdf(pdf_url, save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 500

    # Step 2: Load and process the PDF
    try:
        load_data(save_path, "1")
    except Exception as e:
        return jsonify({"error": f"Failed to load PDF data: {str(e)}"}), 500

    # Step 3: Perform job description and resume analysis
    job = """
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

    prompt = f"""
    Extract and Analyze Features
    Input: {job}
    ...
    """
    try:
        response = get_bot_response(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500


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
