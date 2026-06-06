from langchain_openai import ChatOpenAI

from env_utils import API_KEY, BASE_URL

gpt = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.4
)