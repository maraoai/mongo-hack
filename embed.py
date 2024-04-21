from pathlib import Path
from fastapi import Depends, HTTPException, status
from fastapi.responses import FileResponse
from modal import Stub, web_endpoint, Image, Secret, method, NetworkFileSystem
import os
from typing import Dict
import json

from starlette.requests import Request

APP_NAME = "mongo-hack"

stub = Stub(name=APP_NAME)
RESULTS_DIR = "/results"
results_volume = NetworkFileSystem.from_name(f"{APP_NAME}-results-vol", create_if_missing=True)

image = (
    Image.debian_slim(
        python_version="3.10"
    ).pip_install(
        "replicate",
        "python-jose",
        "requests",
        "openai",
        "pymongo",
        "scikit-learn",
        "numpy"
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
    
    from pymongo import MongoClient
    client = MongoClient("mongodb+srv://cameron:testing123@mongo-hackathon.4lo2sdu.mongodb.net/")
    db = client["mongo-hackathon"]
    print(db)

    collection = db["documents"]
    input_text=payload["text"]

    import openai
    fireworks_client = openai.OpenAI(
        api_key=os.getenv('FIREWORKS_API_KEY'),
        base_url="https://api.fireworks.ai/inference/v1"
    )

    # Check if input text is already in the database
    existing_documents = [d for d in collection.find({"text": input_text})]
    document_exists = len(existing_documents) != 0

    # If we already have the document, get the embedding.
    if document_exists:
        embedding = existing_documents[0]["embedding"]
    else:
        response = fireworks_client.embeddings.create(
            model="nomic-ai/nomic-embed-text-v1.5",
            input=input_text,
            dimensions=768,
        )

        embedding = response.data[0].embedding
    
    print(len(embedding))

    # Okay -- if our document exists, check if it has a title, and if it does, use it. Otherwise, generate a title.
    if document_exists:
        if "title" in existing_documents[0] and existing_documents[0]["title"] != "":
            new_title = existing_documents[0]["title"]
        else:
            new_title = fire_works_create_title(input_text)

            # Update the document
            collection.update_one({"_id": existing_documents[0]["_id"]}, {"$set": {"title": new_title}})
    else:
        new_title = fire_works_create_title(input_text)

    document = {
        # "_id": ObjectId("6624177242be2e1d80040517"),
        "embedding": embedding,
        "text": input_text,
        "title": new_title
    }

    if not document_exists:
        result = collection.insert_one(document)
        print("Inserted document ID:", result.inserted_id)

    def query_results(query_vec):
        results = collection.aggregate([
            {
                '$vectorSearch': {
                    "index": "embedding_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 100,
                    "limit": 100,
                }
            }
            ])
        return results

    # run pipeline
    results = query_results(embedding)
    results_collected = [r for r in results]
    print("reslts collected shape")
    print(len(results_collected))

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    
    def smush(matrix):
        pca = PCA(n_components=2)
        print(matrix.shape)
        smushed = pca.fit_transform(matrix)
        return smushed

    # Converts a matrix to a JSON
    def bundle(texts, titles, matrix):
        smushed = smush(matrix)

        # Normalize the matrix
        scaler = MinMaxScaler(feature_range=(-1, 1))
        smushed = scaler.fit_transform(matrix)
        smushed = smushed - smushed[0,:]

        return {
            "data": [
                {
                    "x": x,
                    "y": y,
                    "text": text,
                    "title": title
                }
                for x, y, text, title in zip(smushed[:, 0], smushed[:, 1], texts, titles)
            ]
        }

    # Unpack things
    texts = [t["text"] for t in results_collected]
    embeddings = [t["embedding"] for t in results_collected]

    print("texts length")
    print(len(texts))

    print("embeddings length")
    print(len(embeddings))

    matrix = np.array(embeddings) - embedding
    return json.dumps(bundle(texts, [""] * len(texts), matrix))


def fire_function_call(client):

    messages = [
        {
        "role": "system",
        "content": "You are a very creative assistant with access to functions. Use them if required."
        },
        {
        "role": "user",
        "content": "Select a super random topic and create a very short title and super detailed long text description for the topic"
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
    
    print(repr(chat_completion.choices[0].message.model_dump()))

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


def fire_function__call_generate_title(client, text):

    messages = [
        {
        "role": "system",
        "content": "You are an assistant good at creative titles with access to functions. Use them if required."
        },
        {
        "role": "user",
        "content": f"Create a very short title for the input text: {text}"
        }
        ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_title",
                "description": "Generate a short title.",
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