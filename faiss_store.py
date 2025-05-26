from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


def store_text_in_faiss(text: str, embedding_size: int = 1536):
    """
    Split the text and store it in a FAISS vectorstore using fake embeddings (for demo/testing).
    Returns the FAISS vectorstore object.
    """
    docs = [text]
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    doc_chunks = splitter.create_documents(docs)
    vectorstore = FAISS.from_documents(
        doc_chunks, FakeEmbeddings(size=embedding_size))
    return vectorstore


def retrieve_relevant_chunk(vectorstore, query: str, k: int = 3) -> str:
    """
    Retrieve the most relevant chunk(s) from the FAISS vectorstore for a given query.
    Returns the page_content of the top chunk.
    """
    relevant_docs = vectorstore.similarity_search(query, k=k)
    return relevant_docs[0].page_content if relevant_docs else ""


if __name__ == "__main__":
    from pathlib import Path
    # Use a real PDF text if available, else a sample
    doc_dir = Path("documents")
    pdfs = list(doc_dir.glob("*.pdf"))
    if pdfs:
        from pdf_text_extractor import extract_text_from_pdf
        with open(pdfs[0], "rb") as f:
            class DummyFile:
                def __init__(self, content):
                    self._content = content

                def read(self):
                    return self._content

                def seek(self, pos):
                    pass
            dummy_file = DummyFile(f.read())
            text = extract_text_from_pdf(dummy_file)
    else:
        text = "This is a test medical bill for John Doe at Test Hospital. Total amount: $1234.56."
    print("[FAISS TEST] Storing text in FAISS...")
    store = store_text_in_faiss(text)
    query = "hospital"
    print(f"[FAISS TEST] Querying for: '{query}'")
    result = retrieve_relevant_chunk(store, query)
    print(f"[FAISS TEST] Top result:\n{result[:300]}\n...")
