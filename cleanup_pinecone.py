from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "chatbot"

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    pc.delete_index(INDEX_NAME)
    print(f"Deleted index '{INDEX_NAME}'. It will be recreated automatically on the next file upload.")
else:
    print(f"Index '{INDEX_NAME}' does not exist — nothing to clean up.")
