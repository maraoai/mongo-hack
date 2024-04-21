# ENV["FIREWORKS_API_KEY"] = "bm38HrH82R4Qc2AdindG4xSHlVb9AlJMcNWyoRAH0akh1Mze"
# ENV["SSL_CERT_DIR"] = "/etc/ssl/certs/"
# ENV["MONGO_CONNSTRING"] = "mongodb+srv://cameron:testing123@mongo-hackathon.4lo2sdu.mongodb.net/?retryWrites=true&w=majority&appName=mongo-hackathon?tlsCAFile=/etc/ssl/certs/ca-certificates.crt"


import os
from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
mongo_connstring = os.environ["MONGO_CONNSTRING"]
client = MongoClient(mongo_connstring)
db = client.data

@app.route('/text', methods=['POST'])
def handle_text():
    # Get the text from the request
    text = request.json['text']
    
    # TODO: Process the text and generate x and y
    x = "example_x"  
    y = "example_y"
    
    # Return the response as JSON
    response = {
        "documents": text,
        "x": x, 
        "y": y
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
