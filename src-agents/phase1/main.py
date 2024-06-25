import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

app = FastAPI()

load_dotenv()

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_false = "true_or_false"
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
        azure_ad_token_provider=token_provider,
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")

@app.get("/")
async def root():
    return {"message": "Hello Smoorghs"}

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

    # Send a completion call to generate an answer
    print(f"""Question: {ask.question}, Type: {ask.type}""")
    response = client.chat.completions.create(
        model = deployment_name,
        messages = [{ "role" : "system", "content" : 
"""You are a question answering bot. Answer the question correctly. For multiple choice, do NOT include the index of the answer (so e.g. instead of "1) Blue" just "Blue").

Examples:
Question: Which movie features a plot where a girl named Dorothy is transported to a magical land? 1) Cinderella 2) The Wizard of Oz
Type: multiple_choice
Answer: The Wizard of Oz

Question: Is Yoda a character from the Star Trek universe: True or False?
Type: true_or_false
Answer: false

Question: How many movies are there in 'The Lord of the Rings'?
Type: estimation
Answer: 3"""},
                      {"role": "user", "content":
f"""Question: {ask.question}
Type: {ask.type}
Answer: """}],
    )

    print(response.choices[0].message.content)
    print(response)
    answer = Answer(answer=response.choices[0].message.content)
    answer.correlationToken = ask.correlationToken
    answer.promptTokensUsed = response.usage.prompt_tokens
    answer.completionTokensUsed = response.usage.completion_tokens

    return answer