using HTTP
using JSON3
using Parquet2, DataFrames
# df = DataFrame(read_parquet(path))

function load_parquet_files(folder_path::String)
    # load the parquet files
    parquet_files = filter(
        x -> endswith(x, ".parquet"),
        readdir(folder_path, join=true)
    )

    # Load and concatenate all parquet files into a single DataFrame
    dfs = DataFrame[]
    for file in parquet_files[1:10]
        ds = Parquet2.Dataset(file)
        df = DataFrame(ds; copycols=false)
        push!(dfs, df)
    end
    return reduce(vcat, dfs)
end

function post_text_to_endpoint(df::DataFrame, url::String)
    # Iterate over each row in the DataFrame
    for row in eachrow(df)
        # Extract text from the current row
        text = row["text"]
        # Prepare the data for POST request
        data = JSON3.write(Dict("text" => text))
        headers = Dict("Content-Type" => "application/json")
        # Send POST request to the specified URL
        response = HTTP.post(url, headers=headers, body=data)
        # Check if the request was successful
        if response.status == 200
            println("Successfully posted text from row")
        else
            println("Failed to post text from row. Status code: $(response.status)")
        end
    end
end

# Define the folder path and URL
folder_path = joinpath(homedir(), "code/tiny-strange-textbooks")
url = "https://cpfiffer--mongo-hack-handle-request-dev.modal.run/"

# Load parquet files into a DataFrame
df = load_parquet_files(folder_path)

# Post text to the specified endpoint
post_text_to_endpoint(df, url)
