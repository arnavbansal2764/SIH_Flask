import requests
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import ollama
from langchain_community.document_loaders import PyMuPDFLoader

client = chromadb.Client()
try:
    collection = client.create_collection(name="docs")
except chromadb.db.base.UniqueConstraintError:
    collection = client.get_collection(name="docs")

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

    print("Loaded")

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

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, 'resume.pdf')
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path
    else:
        return None

def analyze_resume(job_description, resume_url):
    resume_file_path = download_file(resume_url)
    if not resume_file_path:
        return {"error": "Failed to download resume file"}

    load_data(resume_file_path, "1")
    recommendation_response = get_bot_response(job_description)
    return {"recommendation": recommendation_response}
