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
results_volume = NetworkFileSystem.new().persisted(f"{APP_NAME}-results-vol")

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
@web_endpoint(method="POST")
def handle_request(
    payload: Dict,
    request: Request
):

    import openai
    client = openai.OpenAI(
        api_key=os.getenv('FIREWORKS_API_KEY'),
        base_url="https://api.fireworks.ai/inference/v1"
        )

    input_text=payload["text"]

    summary_response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v2-7b-chat",
        messages=[{
            "role": "user",
            "content": f"create a funny title for the given text: {input_text}",
        }],
        )
    print(summary_response.choices[0].message.content)

    response = client.embeddings.create(
        model="nomic-ai/nomic-embed-text-v1.5",
        input=input_text,
        dimensions=768,
    )
    print(len(response.data[0].embedding))

    # And also go make a title

    new_title = "" # T

    from pymongo import MongoClient
    client = MongoClient("mongodb+srv://cameron:testing123@mongo-hackathon.4lo2sdu.mongodb.net/")
    db = client["mongo-hackathon"]
    print(db)

    collection = db["documents"]


    document = {
        # "_id": ObjectId("6624177242be2e1d80040517"),
        "embedding": response.data[0].embedding,
        "text": input_text,
        "title": new_title
        }
    result = collection.insert_one(document)
    print("Inserted document ID:", result.inserted_id)
    

    # Please return all the crap that goes along with this one
    # {
    #   "$vectorSearch": {
    #     "index": "<index-name>",
    #     "path": "<field-to-search>",
    #     "queryVector": [<array-of-numbers>],
    #     "numCandidates": <number-of-candidates>,
    #     "limit": <number-of-results>,
    #     "filter": {<filter-specification>}
    #   }
    # }

    # pipeline = [
    #     {
    #         "$vectorSearch": {
    #             "index": "vector_index",
    #             "vectorSearchQuery": {
    #                 "path": "plot_embedding",
    #                 "queryVector": response.data[0].embedding,
    #                 "numCandidates": 1000,
    #             }
    #         }
    #     },
    #     {
    #         "$limit": 10
    #     }
    # ]

    # # # define pipeline
    # pipeline = [
    # {
    #     '$vectorSearch': {
    #     'index': 'vector_index', 
    #     'path': 'plot_embedding', 
    #     'queryVector': response.data[0].embedding,
    #     'numCandidates': 1000, 
    #     'limit': 10
    #     }
    # }, {
    #     '$project': {
    #         '_id': 1, 
    #         'embedding': 1,
    #         'text': 1, 
    #         'score': {
    #             '$meta': 'vectorSearchScore'
    #         }
    #     }
    # }
    # ]

    def query_results(query_vec):
        results = collection.aggregate([
            {
                '$vectorSearch': {
                    "index": "embedding_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 1000,
                    "limit": 100,
                }
            }
            ])
        return results

    # run pipeline
    # result = collection.aggregate(pipeline)
    results = query_results(response.data[0].embedding)
    results_collected = [r for r in results]
    print("reslts collected shape")
    print(len(results_collected))

    import numpy as np
    from sklearn.decomposition import PCA

    def smush(matrix):
        pca = PCA(n_components=2)
        print(matrix.shape)
        smushed = pca.fit_transform(matrix)
        return smushed

    # Converts a matrix to a JSON with the format
    # {
    #     "data": [
    #         {
    #             "x": x1,
    #             "y": y1,
    #             "text": text1
    #         }
    #         ...
    #     ]
    # }
    def bundle(texts, matrix):
        smushed = smush(matrix)
        return {
            "data": [
                {
                    "x": x,
                    "y": y,
                    "text": text
                }
                for x, y, text in zip(smushed[:, 0], smushed[:, 1], texts)
            ]
        }

    # Unpack things
    texts = [t["text"] for t in results_collected]
    embeddings = [t["embedding"] for t in results_collected]

    print("texts length")
    print(len(texts))

    print("embeddings length")
    print(len(embeddings))

    matrix = np.array(embeddings) - response.data[0].embedding

    # smushed = smush(matrix)
    # return json.dumps({
    #     'xs':smushed[:,0],
    #     'ys':smushed[:,1],
    # })

    return json.dumps(bundle(texts, matrix))
    # return len(response.data[0].embedding)


