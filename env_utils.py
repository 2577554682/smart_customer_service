import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH")