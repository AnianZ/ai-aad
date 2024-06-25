import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizedQuery
)

app = FastAPI()

load_dotenv()

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_or_false = "true_or_false"
    popular_choice = "popular_choice"
    estimation = "estimation"

class Ask(BaseModel):
    question: str | None = None
    type: QuestionType
    correlationToken: str | None = None

class Answer(BaseModel):
    answer: str
    correlationToken: str | None = None
    promptTokensUsed: int | None = None
    completionTokensUsed: int | None = None

client: AzureOpenAI

if "AZURE_OPENAI_API_KEY" in os.environ:
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_ad_token_provider = token_provider,
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
index_name = "movies-semantic-index"
service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

# use an embeddingsmodel to create embeddings
def get_embedding(text, model=embedding_model):
    return client.embeddings.create(input = [text], model=model, dimensions=1536).data[0].embedding

credential = None
if "AZURE_AI_SEARCH_KEY" in os.environ:
    credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"])
else:
    credential = DefaultAzureCredential()

search_client = SearchClient(
    service_endpoint, 
    index_name, 
    credential
)

@app.get("/")
async def root():
    return {"message": "Hello Smorgs"}

@app.get("/healthz", summary="Health check", operation_id="healthz")
async def get_products(query: str = None):
    """
    Returns a status of the app
    """
@app.post("/ask", summary="Ask a question", operation_id="ask") 
async def ask_question(ask: Ask):
    """
    Ask a question
    """
    print('----------------------------------------------------')
    print(f"""Question: {ask.question}, Type: {ask.type}""")


    vector = VectorizedQuery(vector=get_embedding(ask.question), k_nearest_neighbors=5, fields="vector")

    found_docs = list(search_client.search(
        search_text=None,
        query_type="semantic",
        semantic_configuration_name="movies-semantic-config",
        vector_queries=[vector],
        select=["title", "genre", "plot", "year", "rating"],
        top=10
    ))

    results = ""
    for i, doc in enumerate(found_docs, 1):
        results += f'{i}. Title: {doc["title"]}, Genre: {doc["genre"]}, Plot: {doc["plot"]}, Year: {doc["year"]}, Rating: {doc["rating"]}\n'

    print('Search results:')
    print(results)

    system_prompt = ""

    if ask.type == QuestionType.multiple_choice:
        print('Multiple choice question')
        system_prompt ="""You are a question answering bot. Answer the question exclusively based on the context provided below. Do NOT include the index of the answer (so e.g. instead of "1) Blue" just "Blue"). 

# Examples for answer format:
Question: Which movie features a plot where a girl named Dorothy is transported to a magical land? 1) Cinderella 2) The Wizard of Oz
Answer: The Wizard of Oz
"""

    elif ask.type == QuestionType.true_or_false:
        print('True or false question')
        system_prompt ="""You are a question answering bot. Answer the question exclusively based on the context provided below with either True or False.

# Examples for answer format:
Question: Is Yoda a character from the Star Trek universe: True or False?
Answer: false"""

    elif ask.type == QuestionType.estimation:
        print('Estimation question')
        system_prompt ="""You are a question answering bot. Answer the question exclusively based on the context provided below with only the number, no unit or other words. Give the shortest possible answer.

# Examples for answer format:
Question: How many movies are there in 'The Lord of the Rings'?
Answer: 3"""

    parameters = [system_prompt, '\n# Context:\n', results, '\n# Question:\n', ask.question]
    joined_parameters = ''.join(parameters)

    response = client.chat.completions.create(
        model = deployment_name,
        messages = [{"role" : "system", "content" : joined_parameters}, {"role": "assistant", "content": "Answer: "}],
    )
    
    print("Answer: ", response.choices[0].message.content)
    answer = Answer(answer=response.choices[0].message.content)
    answer.correlationToken = ask.correlationToken
    answer.promptTokensUsed = response.usage.prompt_tokens
    answer.completionTokensUsed = response.usage.completion_tokens

    return answer