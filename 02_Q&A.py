# Step 1 : Loading libraries and setting up utility functions
import json
import os
from dotenv import load_dotenv

# For cost-saving, create a cache for the LLM responses
import threading

# For data analysis and visualization
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd

# For scraping
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

load_dotenv()

class Cache:
    def __init__(self, persist_path, cache_loading_fn):
        """
        The cache_loading_fn should be a function that takes arbitrary
        serializable arguments and returns a serilaizable value.
          value = cache_loading_fn(**kwargs)
        For example, for openai.chat.completions.create(...), the
        cache_loading_fn should be:
          def cache_loading_fn(**kwargs):
            result = openai.chat.completions.create(**kwargs)
            return result.to_dict_recursive()
        """
        self._cache = self._get_or_create_cache_dict(persist_path)
        self._persist_path = persist_path
        self._cache_loading_fn = cache_loading_fn
        self._cache_lock = threading.Lock()

    @classmethod
    def _get_or_create_cache_dict(cls, persist_path):
        if os.path.exists(persist_path):
            # File exists, load it as a JSON string into a dict
            with open(persist_path) as f:
                cache = json.load(f)
        else:
            # File does not exist, create an empty dict
            cache = {}
        return cache

    def _save_to_file(self):
        with open(self._persist_path, "w") as file:
            json.dump(self._cache, file)

    def _update_cache(self, key, value):
        with self._cache_lock:
            self._cache[key] = value
            self._save_to_file()

    def get_from_cache_or_load_cache(self, **kwargs):
        key = json.dumps(kwargs)

        with self._cache_lock:
            value = self._cache.get(key, None)

        if value is None:
            value = self._cache_loading_fn(**kwargs)
            self._update_cache(key, value)
        else:
            print("Loaded from cache")

        return value


def chat_completion_create_fn(**kwargs):
    result = openai.chat.completions.create(**kwargs)
    return result.to_dict_recursive()


def cached_openai_ChatCompletion_create(**kwargs):
    cache = kwargs.pop("cache")
    return cache.get_from_cache_or_load_cache(**kwargs)


def embeddings_embed_documents_fn(**kwargs):
    chunk = kwargs.get("chunk")
    return embeddings.embed_documents([chunk])


def cached_langchain_openai_embeddings(**kwargs):
    cache = kwargs.pop("cache")
    return cache.get_from_cache_or_load_cache(**kwargs)

# Step 2 : Set OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai.api_key

# Other configurations

# Choose a seed for reproducible results
SEED = 2023

# For cost-saving purposes, choose a path to persist the responses for LLM calls
CACHE_PATH = "./data/_cache.json"
EMBEDDINGS_CACHE_PATH = "./data/_embeddings_cache.json"

# To avoid re-running the scraping process, choose a path to save the scrapped docs
SCRAPPED_DATA_PATH = "./data/mlflow_docs_scraped.csv"

# Choose a path to save the generated dataset
OUTPUT_DF_PATH = "./data/question_answer_source.csv"

cache = Cache(CACHE_PATH, chat_completion_create_fn)
embeddings_cache = Cache(EMBEDDINGS_CACHE_PATH, embeddings_embed_documents_fn)

# Step 3: Decide on Chunk Size
CHUNK_SIZE = 1500

# Step 4: Prepare Document Data
# Scrape the documents from the MLflow website
page = requests.get("https://mlflow.org/docs/latest/index.html")
soup = BeautifulSoup(page.content, "html.parser")

mainLocation = "https://mlflow.org/docs/latest/"
header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

data = []
for a_link in soup.find_all("a"):
    document_url = mainLocation + a_link["href"]
    page = requests.get(document_url, headers=header)
    soup = BeautifulSoup(page.content, "html.parser")
    file_to_store = a_link.get("href")
    if soup.find("div", {"class": "rst-content"}):
        data.append(
            [
                file_to_store,
                soup.find("div", {"class": "rst-content"}).text.replace("\n", " "),
            ]
        )

df = pd.DataFrame(data, columns=["source", "text"])

df.to_csv(SCRAPPED_DATA_PATH, index=False)
df = pd.read_csv(SCRAPPED_DATA_PATH)

# Select a subset of the documents and split them into chunks
# For demonstration purposes, let's pick 5 popular MLflow documantation pages from the dataset
mask = df["source"].isin(
    {
        "tracking.html",
        "models.html",
        "model-registry.html",
        "search-runs.html",
        "projects.html",
    }
)
sub_df = df[mask]

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, separator=" ")


def get_chunks(input_row):
    new_rows = []
    chunks = text_splitter.split_text(input_row["text"])
    for i, chunk in enumerate(chunks):
        new_rows.append({"chunk": chunk, "source": input_row["source"], "chunk_index": i})
    return new_rows


expanded_df = pd.DataFrame(columns=["chunk", "source", "chunk_index"])

for index, row in sub_df.iterrows():
    new_rows = get_chunks(row)
    expanded_df = pd.concat([expanded_df, pd.DataFrame(new_rows)], ignore_index=True)

expanded_df.head(3)

# For cost-saving purposes, let's pick the first 3 chunks from each doc
# To generate questions with more chunks, change the start index and end index in iloc[]
start, end = 0, 3
filtered_df = (
    expanded_df.groupby("source").apply(lambda x: x.iloc[start:end]).reset_index(drop=True)
)
filtered_df.head(3)

filtered_df["chunk"][0]

# Step 5: Generate questions
def get_raw_response(content):
    prompt = f"""Please generate a question asking for the key information in the given paragraph.
    Also answer the questions using the information in the given paragraph.
    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'.
    Please generate the answer using as much information as possible.
    If you are unable to answer it, please generate the answer as 'I don't know.'
    The answer should be informative and should be more than 3 sentences.

    Paragraph: {content}

    Please call the submit_function function to submit the generated question and answer.
    """

    messages = [{"role": "user", "content": prompt}]

    submit_function = {
        "name": "submit_function",
        "description": "Call this function to submit the generated question and answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asking for the key information in the given paragraph.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to the question using the information in the given paragraph.",
                },
            },
            "required": ["question", "answer"],
        },
    }

    return cached_openai_ChatCompletion_create(
        messages=messages,
        model="gpt-4o-mini",
        functions=[submit_function],
        function_call="auto",
        temperature=0.0,
        seed=SEED,
        cache=cache,
    )


def generate_question_answer(content):
    if content is None or len(content) == 0:
        return "", "N/A"

    response = get_raw_response(content)
    try:
        func_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        question = func_args["question"]
        answer = func_args["answer"]
        return question, answer
    except Exception as e:
        return str(e), "N/A"

queries = []

get_raw_response(filtered_df["chunk"][0])

# The requests sometimes get ratelimited, you can re-execute this cell without losing the existing results.
n = len(filtered_df)
for i, row in filtered_df.iterrows():
    chunk = row["chunk"]
    question, answer = generate_question_answer(chunk)
    print(f"{i+1}/{n}: {question}")
    queries.append(
        {
            "question": question,
            "answer": answer,
            "chunk": chunk,
            "chunk_id": row["chunk_index"],
            "source": row["source"],
        }
    )

result_df = pd.DataFrame(queries)
result_df = result_df[result_df["answer"] != "N/A"]

def add_to_output_df(result_df=pd.DataFrame({})):
    """
    This function adds the records in result_df to the existing records saved at OUTPUT_DF_PATH,
    remove the duplicate rows and save the new collection of records back to OUTPUT_DF_PATH.
    """
    if os.path.exists(OUTPUT_DF_PATH):
        all_result_df = pd.read_csv(OUTPUT_DF_PATH)
    else:
        all_result_df = pd.DataFrame({})
    all_result_df = (
        pd.concat([all_result_df, result_df], ignore_index=True)
        .drop_duplicates()
        .sort_values(by=["source", "chunk_id"])
        .reset_index(drop=True)
    )
    all_result_df.to_csv(OUTPUT_DF_PATH, index=False)
    return all_result_df

all_result_df = add_to_output_df(result_df)

all_result_df.head(3)