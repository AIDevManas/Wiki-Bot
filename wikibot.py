import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import retrieval_qa


def setup_user_agent():
    ## Set a custom User-Agent to avoid being blocked by websites
    if "USER_AGENT" not in os.environ:
        os.environ["USER_AGENT"] = "wikibot/1.0 (+https://example.com/bot-info)"
        print("Setting User-Agent")


def load_wikipedia_page():
    ## Load a Wikipedia page using WebBaseLoader
    loader = WebBaseLoader(
        web_paths=("https://en.wikipedia.org/wiki/Large_language_model",),
    )
    docs = loader.load()  # Print the first 500 characters of the page content
    print(f"Fetch {len(docs[0].page_content)} characters")
    return docs


def process_page_content(docs):
    ## Process the page content with BeautifulSoup
    soup = bs4.BeautifulSoup(docs[0].page_content, "html.parser")
    text = soup.get_text()
    return text


def split_text(text):
    ## Split the text into manageable chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    print(texts[0])
    print(f"Split into {len(texts)} chunk")
    return texts


def embed_and_store(texts):
    ## Embed the text chunks and store them in a FAISS vector store

    load_dotenv(dotenv_path=".env")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.add_texts(texts)
    vector_store.save_local("faiss_index")
    print("FAISS index saved locally")
    return vector_store


def setup_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    qa_chain = retrieval_qa.RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        ),
        chain_type="stuff",
    )

    return qa_chain


def main():
    setup_user_agent()
    docs = load_wikipedia_page()
    text = process_page_content(docs)
    texts = split_text(text)
    vector_store = embed_and_store(texts)
    qa_chain = setup_qa_chain(vector_store)
    ans = qa_chain.invoke({"query": "What is a large language model?"})
    print("Answer:", ans["result"])


if __name__ == "__main__":
    main()
