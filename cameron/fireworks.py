import openai

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key="bm38HrH82R4Qc2AdindG4xSHlVb9AlJMcNWyoRAH0akh1Mze",
)
response = client.embeddings.create(
  model="nomic-ai/nomic-embed-text-v1.5",
  input="search_document: Spiderman was a particularly entertaining movie with...",
)

print(response)