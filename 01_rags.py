# Loading libraries
import ast
import os
from IPython import display
import mlflow
from dotenv import load_dotenv
import pandas as pd

import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
rag_experiment = mlflow.set_experiment("rag_experiment")
run_name = "rag_test"

loader = WebBaseLoader(
    [
        "https://mlflow.org/docs/latest/index.html",
        "https://mlflow.org/docs/latest/tracking/autolog.html",
        "https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
        "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html",
    ]
)

documents = loader.load()
CHUNK_SIZE = 1000
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

llm = ChatOpenAI(model='gpt-4o-mini',
                 temperature=0.1,
                 top_p=0.1,
                 max_tokens=500,
                 openai_api_key = openai_api_key 
                 )

embedding_function = OpenAIEmbeddings(model = 'text-embedding-3-small')
docsearch = Chroma.from_documents(texts, embedding_function)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(fetch_k=3),
    return_source_documents=True,
)

EVALUATION_DATASET_PATH = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/llms/RAG/static_evaluation_dataset.csv"

synthetic_eval_data = pd.read_csv(EVALUATION_DATASET_PATH)

# Load the static evaluation dataset from disk and deserialize the source and retrieved doc ids
synthetic_eval_data["source"] = synthetic_eval_data["source"].apply(ast.literal_eval)
synthetic_eval_data["retrieved_doc_ids"] = synthetic_eval_data["retrieved_doc_ids"].apply(
    ast.literal_eval
)

eval_data = pd.DataFrame(
    {
        "question": [
            "What is MLflow?",
            "What is Databricks?",
            "How to serve a model on Databricks?",
            "How to enable MLflow Autologging for my workspace by default?",
        ],
        "source": [
            ["https://mlflow.org/docs/latest/index.html"],
            ["https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html"],
            ["https://mlflow.org/docs/latest/python_api/mlflow.deployments.html"],
            ["https://mlflow.org/docs/latest/tracking/autolog.html"],
        ],
    }
)

def evaluate_embedding(embedding_function):
    CHUNK_SIZE = 1000
    list_of_documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    docs = text_splitter.split_documents(list_of_documents)
    retriever = Chroma.from_documents(docs, embedding_function).as_retriever()

    def retrieve_doc_ids(question: str) -> list[str]:
        docs = retriever.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]

    def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
        return question_df["question"].apply(retrieve_doc_ids)

    with mlflow.start_run(run_name=run_name):
        return mlflow.evaluate(
            model=retriever_model_function,
            data=eval_data,
            model_type="retriever",
            targets="source",
            evaluators="default",
        )


result1 = evaluate_embedding(embedding_function)
# To validate the results of a different model, comment out the above line and uncomment the below line:
# result2 = evaluate_embedding(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

eval_results_of_retriever_df_bge = result1.tables["eval_results_table"]
# To validate the results of a different model, comment out the above line and uncomment the below line:
# eval_results_of_retriever_df_MiniLM = result2.tables["eval_results_table"]
print(eval_results_of_retriever_df_bge)