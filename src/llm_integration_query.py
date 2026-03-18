import os
from pathlib import Path
from typing import cast

from decouple import config
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI

# Define the storage directory
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "storage"
DATA_FILE = BASE_DIR / "data" / "pep8.rst"


def get_index(persist_dir=PERSIST_DIR, data_file=DATA_FILE):
    os.environ["OPENAI_API_KEY"] = cast(str, config("OPENAI_API_KEY"))

    if persist_dir.exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
        print("Index loaded from storage...")
    else:
        reader = SimpleDirectoryReader(input_files=[str(data_file)])
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print("Index created and persisted to storage...")

    return index


def main():
    index = get_index()
    llm = OpenAI(model="gpt-5.1")
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("Summarize the import rules.")
    print(response)


if __name__ == "__main__":
    main()
