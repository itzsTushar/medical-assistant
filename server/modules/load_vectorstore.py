import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone,ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
load_dotenv()
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medicalstindex"


#os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
UPLOAD_DIR = "./upload_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)

#initlize pinecone instace
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)
        
index = pc.Index(PINECONE_INDEX_NAME)
#load,split,embed,and upsert pdf docs
def load_vectorstore(uploaded_files):
    #embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    file_paths=[]
    #upload
    for file in uploaded_files:
        save_paths=Path(UPLOAD_DIR)/file.filename
        with open(save_paths,"wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_paths))
    
    #split
    for file_path in file_paths:
        loader=PyPDFLoader(file_path)
        documents=loader.load()
        
        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        chunks=splitter.split_documents(documents)
        
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i, _ in enumerate(chunks)]

        #Embeddings
        print(f"Embedding chunks")
        embedding = embed_model.encode(texts)
        
        #upsert
        print("Upserting Embeddings (batch-wise)...")

        for i in tqdm(range(0, len(embedding), 50), desc="Upserting to Pinecone"):
            batch_vectors = list(zip(
                ids[i:i+50],
                embedding[i:i+50],
                metadata[i:i+50]
            ))
        index.upsert(vectors=batch_vectors)    
        print(f"Upload Complete for {file_path}")

