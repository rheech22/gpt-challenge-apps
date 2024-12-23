from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path
import streamlit as st

CLOUDFLARE_SITEMAP_URL = 'https://developers.cloudflare.com/sitemap-0.xml'

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    llm.streaming = True
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

@st.cache_data(show_spinner="Loading data...")
def load_docs(api_key): 
    loader = SitemapLoader(
        CLOUDFLARE_SITEMAP_URL,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = loader.load_and_split(text_splitter=splitter)

    Path("./.cache/embeddings").mkdir(parents=True, exist_ok=True)
    cache_dir = LocalFileStore(f"./.cache/embeddings/cloudflare")
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

with st.sidebar:
    api_key = st.text_input(
        "Write down your OpenAI API Key",
        placeholder="sh-123213123123123123123",
    )

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

if api_key:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )
    retriever = load_docs(api_key)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    query = st.chat_input("Ask a question about Cloudflare...")
    if query:
        send_message(query, "human")
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            response = chain.invoke(query)
else:
    st.markdown(
        """
    Welcome!
                
    Use this chatbot to ask questions to an AI about Cloudflare!

    1. Input your OpenAI API Key on the sidebar
    2. Ask questions related to the Cloudflare.
    """
    )
    st.session_state["messages"] = []