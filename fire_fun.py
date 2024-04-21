from pathlib import Path
from fastapi import Depends, HTTPException, status
from fastapi.responses import FileResponse
from modal import Stub, web_endpoint, Image, Secret, method, NetworkFileSystem
import os
from typing import Dict
import json

from starlette.requests import Request

APP_NAME = "mongo-hack-fire-fun"

stub = Stub(name=APP_NAME)
RESULTS_DIR = "/results"
results_volume = NetworkFileSystem.from_name(f"{APP_NAME}-results-vol", create_if_missing=True)

image = (
    Image.debian_slim(
        python_version="3.11"
    ).pip_install(
        "replicate",
        "python-jose",
        "requests",
        "openai",
        "scikit-learn",
        "numpy",
        "pymongo"
    )
)

@stub.function(image=image, keep_warm=1, secrets=[Secret.from_name("mongo-hack-secret")])
@web_endpoint(method="GET")
def fetch_doc_text(
    docId: str,
    payload: Dict,
    request: Request
):

    print(docId)

    from pymongo import MongoClient
    from bson.objectid import ObjectId


    # Connect to MongoDB Atlas
    client = MongoClient("mongodb+srv://cameron:testing123@mongo-hackathon.4lo2sdu.mongodb.net/")
    db = client["mongo-hackathon"]
    collection = db["documents"]

    print(collection)

    # Specify the document ID
    document_id = docId

    # Find the document by its ID
    document = collection.find_one({'_id': ObjectId(document_id)})

    if document:
        print('Document found:')
        return document['text']
    else:
        print('Document not found.')
        return document['Document not found.']



@stub.function(image=image, keep_warm=1, secrets=[Secret.from_name("mongo-hack-secret")])
@web_endpoint(method="POST")
def handle_request(
    payload: Dict,
    request: Request
):
    import openai

    client = openai.OpenAI(
        base_url = "https://api.fireworks.ai/inference/v1",
        api_key = os.getenv('FIREWORKS_API_KEY')
    )

    fire_function__call_generate_title(client, "James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? Let's be accurate as possible.")

    fire_function_call(client)



def fire_function_call(client):

    messages = [
        {
        "role": "system",
        "content": "You are a very creative assistant with access to functions. Use them if required."
        },
        {
        "role": "user",
        "content": "Select a super random topic and create a title and super detailed long text description for the topic"
        }
        ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_title_text",
                "description": "Get the random text and title.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "random_text": {
                            "type": "string",
                            "description": "random text",
                        },
                        "title": {
                            "type": "string",
                            "description": "title for random text."
                        },
                    },
                    "required": ["random_text", "title"],
                },
            },
        }
    ]
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/fw-function-call-34b-v0",
        messages=messages,
        tools=tools,
    tool_choice="auto",
    temperature=0.8
    )
    
    # print(repr(chat_completion.choices[0].message.model_dump()))

    message_data = chat_completion.choices[0].message.model_dump()

    # Assuming message_data now contains a dictionary with the 'tool_calls' key
    tool_calls = message_data['tool_calls']

    # Extract the function arguments from the first tool call
    function_args_str = tool_calls[0]['function']['arguments']

    # Parse the JSON string in function arguments
    function_args = json.loads(function_args_str)

    # Extract 'random_text' and 'title' values
    random_text = function_args['random_text']
    title = function_args['title']

    print("Random Text:", random_text)
    print("Title:", title)

    return random_text, title


def fire_function__call_generate_title(client, text):

    messages = [
        {
        "role": "system",
        "content": "You are an assistant good at creative titles with access to functions. Use them if required."
        },
        {
        "role": "user",
        "content": f"create a very short title for following input text: {text}"
        }
        ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_title",
                "description": "Geenerate title.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "title for input text."
                        },
                    },
                    "required": ["title"],
                },
            },
        }
    ]
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/fw-function-call-34b-v0",
        messages=messages,
        tools=tools,
    tool_choice="auto",
    temperature=0.1
    )
    
    print(repr(chat_completion.choices[0].message.model_dump()))

    message_data = chat_completion.choices[0].message.model_dump()

    # Assuming message_data now contains a dictionary with the 'tool_calls' key
    tool_calls = message_data['tool_calls']

    # Extract the function arguments from the first tool call
    function_args_str = tool_calls[0]['function']['arguments']

    # Parse the JSON string in function arguments
    function_args = json.loads(function_args_str)

    # Extract 'random_text' and 'title' values
    title = function_args['title']

    print("Title:", title)

    return title


    def fire_works_create_title(text):
        client = openai.OpenAI(
            base_url = "https://api.fireworks.ai/inference/v1",
            api_key=os.getenv('FIREWORKS_API_KEY'),
        )
        response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v2-7b-chat",
        messages=[{
            "role": "user",
            "content": f"create a title for this text: {text}",
        }],
        )
        print(response.choices[0].message.content)

    return response.choices[0].message.content


@stub.function(image=image, keep_warm=1, secrets=[Secret.from_name("mongo-hack-secret")])
@web_endpoint(method="GET")
def insert_titles(
):

    from pymongo import MongoClient
    from bson.objectid import ObjectId


    # Connect to MongoDB Atlas
    client = MongoClient("mongodb+srv://cameron:testing123@mongo-hackathon.4lo2sdu.mongodb.net/")
    db = client["mongo-hackathon"]
    collection = db["documents"]

    print(collection)

    # Retrieve all documents in the collection
    documents = collection.find()

    for document in documents:
        # Get the title of the document
        print(document)
        title = document.get("title", "No title available")
        
        # Get the document ID
        document_id = str(document["_id"])
        
        print(f"Document ID: {document_id}")
