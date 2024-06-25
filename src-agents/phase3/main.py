import os
import json
import requests
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
        azure_ad_token_provider=token_provider,
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
index_name = "movies-semantic-index"
service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_COMPLETION_MODEL")

smoorghApi = "https://smoorgh-api.bluebush-897105f3.northeurope.azurecontainerapps.io/"

def get_movie_rating(title):
    try:
        headers = {"title": title}
        response = requests.get(f"{smoorghApi}rating", headers=headers)
        print('The api response for rating is:', response.text)
        return response.text

    except:
        return "Sorry, I couldn't find a rating for that movie."

def get_movie_year(title):
    try:
        headers = {"title": title}
        response = requests.get(f"{smoorghApi}year", headers=headers)
        print('The api response for year is:', response.text)
        return response.text

    except:
        return "Sorry, I couldn't find a year for that movie."
    
def get_movie_actor(title):
    try:
        headers = {"title": title}
        response = requests.get(f"{smoorghApi}actor", headers=headers)
        print('The api response for actor is:', response.text)
        return response.text

    except:
        return "Sorry, I couldn't find a actor for that movie."
    
def get_movie_location(title):
    try:
        headers = {"title": title}
        response = requests.get(f"{smoorghApi}location", headers=headers)
        print('The api response for location is:', response.text)
        return response.text

    except:
        return "Sorry, I couldn't find a actor for that movie."

def get_movie_genre(title):
    try:
        headers = {"title": title}
        response = requests.get(f"{smoorghApi}genre", headers=headers)
        print('The api response for genre is:', response.text)
        return response.text

    except:
        return "Sorry, I couldn't find a actor for that movie."

functions = [
        {
            "type": "function",
            "function": {
                "name": "get_movie_rating",
                "description": "Gets the rating of a movie",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The movie name. The movie name should be a string without quotation marks.",
                        }
                    },
                    "required": ["title"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_movie_location",
                "description": "Gets the location of a movie",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The movie name. The movie name should be a string without quotation marks.",
                        }
                    },
                    "required": ["title"],
                },
            }
        },
        {  
            "type": "function",  
            "function": {  
                "name": "get_movie_year",  
                "description": "Gets the release year of a movie",  
                "parameters": {  
                    "type": "object",  
                    "properties": {  
                        "title": {  
                            "type": "string",  
                            "description": "The movie name. The movie name should be a string without quotation marks."  
                        }  
                    },  
                    "required": ["title"]  
                }  
            }  
        },
        {  
            "type": "function",  
            "function": {  
                "name": "get_movie_actor",  
                "description": "Gets the leading actor of a movie",  
                "parameters": {  
                    "type": "object",  
                    "properties": {  
                        "title": {  
                            "type": "string",  
                            "description": "The movie name. The movie name should be a string without quotation marks."  
                        }  
                    },  
                    "required": ["title"]  
                }  
            }  
        },
        {  
            "type": "function",  
            "function": {  
                "name": "get_movie_genre",  
                "description": "Gets the genre of a movie",  
                "parameters": {  
                    "type": "object",  
                    "properties": {  
                        "title": {  
                            "type": "string",  
                            "description": "The movie name. The movie name should be a string without quotation marks."  
                        }  
                    },  
                    "required": ["title"]  
                }  
            }  
        }  
]  

available_functions = {
            "get_movie_rating": get_movie_rating,
            "get_movie_location": get_movie_location,
            "get_movie_actor": get_movie_actor,
            "get_movie_genre": get_movie_genre,
            "get_movie_year": get_movie_year,
        } 

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
    system_prompt = ""

    if ask.type == QuestionType.multiple_choice:
        print('Multiple choice question')
        system_prompt ="""You are a question answering bot. Use the tools available to you. Do NOT include the index of the answer (so e.g. instead of "1) Blue" just "Blue"). 

# Examples for answer format:
Question: Which movie features a plot where a girl named Dorothy is transported to a magical land? 1) Cinderella 2) The Wizard of Oz
Answer: The Wizard of Oz
"""

    elif ask.type == QuestionType.true_or_false:
        print('True or false question')
        system_prompt ="""You are a question answering bot. Use the tools available to you. Answer the question with either True or False.

# Examples for answer format:
Question: Is Yoda a character from the Star Trek universe: True or False?
Answer: false"""

    elif ask.type == QuestionType.estimation:
        print('Estimation question')
        system_prompt ="""You are a question answering bot. Use the tools available to you. Answer the question with only the number, no unit or other words. Give the shortest possible answer.

# Examples for answer format:
Question: How many movies are there in 'The Lord of the Rings'?
Answer: 3"""

    question = ask.question
    messages= [
               { "role" : "system", "content" : f"""{system_prompt}\n\nQuestion: {question}"""}
               ]
    first_response = client.chat.completions.create(
        model = deployment_name,
        messages = messages,
        tools = functions,
        tool_choice = "auto",
    )

    print(first_response)
    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls

     # Step 2: check if GPT wanted to call a function
    if tool_calls:
        print("Recommended Function call:")
        print(tool_calls)
        print()
    
        # Step 3: call the function
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            # verify function exists
            if function_name not in available_functions:
                return "Function " + function_name + " does not exist"
            else:
                print("Calling function: " + function_name)
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            print(function_args)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            ) 
            print("Addding this message to the next prompt:") 
            print (messages)
            
             # extend conversation with function response
            second_response = client.chat.completions.create(
                model = deployment_name,
                messages = messages)  # get a new response from the model where it can see the function response
            
            print("second_response")
            print(second_response.choices[0].message.content)
            
            answer = Answer(answer=second_response.choices[0].message.content)
            answer.correlationToken = ask.correlationToken
            answer.promptTokensUsed = second_response.usage.prompt_tokens
            answer.completionTokensUsed = second_response.usage.completion_tokens

            return answer
