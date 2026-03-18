import os
from typing import cast

from decouple import config
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def main():
    os.environ["OPENAI_API_KEY"] = cast(str, config("OPENAI_API_KEY"))
    reader = SimpleDirectoryReader(input_files=["./data/pep8.rst"])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print(query_engine.query("What is this document about?"))


if __name__ == "__main__":
    main()
