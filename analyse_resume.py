import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import ollama
from langchain_community.document_loaders import PyMuPDFLoader

# Set up ChromaDB client
client = chromadb.Client()
try:
    collection = client.create_collection(name="docs")
except chromadb.db.base.UniqueConstraintError:
    collection = client.get_collection(name="docs")


# Function to download the file from a URL and return the local path
def download_file(file_url):
    local_filename = file_url.split("/")[-1]
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        return local_filename
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")


# Function to load data and store embeddings
def load_data(file_url, user_id):
    # Download the PDF from the provided URL
    path = download_file(file_url)

    # Load the PDF content
    loader = PyMuPDFLoader(path)
    data = loader.load()

    list1 = []

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    text_splitter.split_documents(data)

    # Prepare data and embed each chunk
    for i in range(len(data)):
        data_text = data[i].page_content
        data_text = data_text.replace("\n", "")
        list1.append(data_text)

    # Add embeddings to ChromaDB collection
    for i, data in enumerate(list1):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=data)
        embedding = response["embedding"]
        collection.add(ids=[str(i) + str(user_id)], embeddings=[embedding], documents=[data])

    print("Resume data loaded")


# Function to generate the response based on the job description and resume data
def get_bot_response(job_description):
    # Embed the job description
    response = ollama.embeddings(prompt=job_description, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    
    # Retrieve the most similar document
    data = results["documents"][0][0]
    print("Data retrieved for analysis")

    # Generate response using the retrieved document
    output = ollama.generate(
        model="wizardlm2",
        prompt=f"Using this data: {data}. Respond to this prompt: {job_description}",
    )
    return output["response"]
