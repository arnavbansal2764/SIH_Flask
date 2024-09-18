from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import ollama

# Initialize ChromaDB client
client = chromadb.Client()

# Try creating or accessing the collection
try:
    collection = client.create_collection(name="resume_collection")
except chromadb.db.base.UniqueConstraintError:
    collection = client.get_collection(name="resume_collection")

# Function to load user data and generate embeddings
def load_user_data(user_info, user_id):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    # Split user information into chunks
    chunks = text_splitter.split_text(user_info)

    # Store embeddings and documents
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=chunk)
        embedding = response["embedding"]
        collection.add(ids=[str(i) + str(user_id)], embeddings=[embedding], documents=[chunk])

    print(f"Loaded data for user {user_id}")

# Function to generate a resume using LLaMA based on user input
def generate_resume(user_input, user_id):
    user_prompt_embedding = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")["embedding"]
    results = collection.query(query_embeddings=[user_prompt_embedding], n_results=1)
    
    best_match = results["documents"][0][0]

    resume_prompt = f"""
    you are a resume builder. Using the following details: {best_match}, generate a professional resume for the user including sections like Education, Experience, Skills, and Summary. 
    User input: {user_input}
    """

    output = ollama.generate(model="llama3.1:8b", prompt=resume_prompt)
    return output["response"]
