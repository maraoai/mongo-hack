using Mongoc
using PromptingTools
using Clustering
using MultivariateStats
using Statistics
using Plots

# get env shit
include("env.jl")

connection_string = ENV["MONGO_CONNSTRING"]
client = Mongoc.Client(connection_string)
@assert Mongoc.ping(client)["ok"] == 1
db = client["mongo-hackathon"]
collection = db["documents"]

# Convenience function to get embedding
function embed_fireworks(text::AbstractString; model="nomic-ai/nomic-embed-text-v1.5")
    return PromptingTools.aiembed(
        PromptingTools.FireworksOpenAISchema(),
        text,
        model=model
    )
end

function embed(text::AbstractString; model="mxbai-embed-large")
    return PromptingTools.aiembed(
        PromptingTools.OllamaSchema(),
        text,
        model=model
    ).content |> Array
end

# Random text generator
function makeprompt()
    # 
    new_prompt = aigenerate(
        PromptingTools.OllamaSchema(),
        "Give me an interesting fact. Please respond only with the fact.",
        model="mistral:latest"
    ).content

    println(new_prompt)

    return new_prompt

    # Ask more
    # response = aigenerate(
    #     PromptingTools.OllamaSchema(),
    #     "Please tell me more about $new_prompt",
    #     model="gemma:2b"
    # ).content

    # # Make a title for this text
    # title = aigenerate(
    #     PromptingTools.OllamaSchema(),
    #     "Please give me a title for this text: $new_prompt",
    #     model="gemma:2b"
    # ).content

    # return (title, new_prompt, response)
end

# bson convenience function
bson(x) = Mongoc.BSON(x)

# Function to upload text to mongo
function upload(text::AbstractString; model="nomic-embed-text:latest")
    # Check if the text already exists
    exists = Mongoc.find_one(collection, bson(Dict("text" => text)))
    if !isnothing(exists)
        @info "Doc exists, skipping insert"
        return exists
    end

    # Get embedding
    embedding = embed(text; model=model)

    # Start a document
    doc = Dict(
        "text" => text,
        "embedding" => embedding
    )

    # Insert into collection
    push!(collection, bson(doc))
end

# upload("cameron")
# upload("gus")
# upload("anand")
# upload("dog")
# upload("cat")

for i in 1:1000
    upload(makeprompt())
end





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

# qvec = "cameron"
# query_embedding = embed(qvec)
# query_doc = bson(Dict(
#     "\$vectorSearch" => Dict(
#         "queryVector" => [query_embedding],
#     )
# ))

# println(query_doc)

# for c in Mongoc.aggregate(collection, query_doc)
#     println(c)
# end

# docs = collect(collection)
embeddings = reduce(hcat, map(identity, doc["embedding"]) for doc in docs)
texts = [doc["text"] for doc in docs]
M = fit(PCA, embeddings; maxoutdim=2)
smushed = predict(M, embeddings)
scanned = Clustering.dbscan(smushed, 0.5)

cluster_docs = Dict{Int,Vector{String}}()
for (assignment, text) = zip(scanned.assignments, texts)
    if !haskey(cluster_docs, assignment)
        cluster_docs[assignment] = String[]
    end
    push!(cluster_docs[assignment], text)
end

# Now we want to transform this matrix so that the first row is the origin
transformation_matrix = smushed

