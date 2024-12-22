import json
import streamlit as st

from pathlib import Path
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

function = {
    "name": "make_quiz",
    "description": "It takes a list of questions, answers and makes a quiz.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    if context language is Korean, you should make a question by Korean.

    There is two different kinds of level, easy and hard.

    The easy level is for the beginner and the hard level is for the advanced.

    if level is hard, you should make a question that is more difficult than easy level.

    The DIFFICULTY of the question is determined by the level of the question.

    The current level is {level}.
         
    Use (o) to signal the correct answer.
         
    Question examples(Easy):
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question examples(Hard):

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: How many countries are in the world?
    Answers: 195|196(o)|197|198
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making Quiz...")
def make_quiz(_docs, topic, level):
    chain = prompt | llm
    return chain.invoke({
        "context": format_docs(_docs),
        "level": level,
    })

@st.cache_data(show_spinner="Making Quiz...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    return retriever.get_relevant_documents(term)

with st.sidebar:
    docs = None
    topic = None
    openai_api_key = st.text_input("Input your OpenAI API Key")

    level = st.selectbox(
        "Choose level of difficulty",
        (
            "Easy",
            "Hard",
        )
    )

    choice = st.selectbox(
        "Choose what u want to use",
        (
            "File",
            "Wikipedia Article",
        )
    )

    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt or .pdf file", type=["docx", "txt", "pdf"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia for a topic")
        if topic:
            docs = wiki_search(topic)


if not (docs and openai_api_key):
    st.markdown(
        """
        This app allows you to generate a quiz from a text document or a Wikipedia article.
        """
    )
else:
    llm = ChatOpenAI(
        temperature=0.1,
        # model_name="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=openai_api_key,
    ).bind(
        function_call={
            "name": "make_quiz",
        },
        functions=[
            function,
        ],
    )
    make_quiz_result = make_quiz(docs, topic if topic else file.name, level)
    quiz = make_quiz_result.additional_kwargs["function_call"]["arguments"]
    with st.form("questions_form"):
        questions = json.loads(quiz)["questions"]
        question_count = len(questions)
        success_count = 0
        for q in questions:
            st.write(q["question"])
            value = st.radio(
                "Select an answer.",
                [a["answer"] for a in q["answers"]],
                index=None
            )
            if { "answer": value, "correct": True } in q["answers"]:
                st.success("Correct answer !")
                success_count += 1
            elif value is not None:
                st.error("Wrong answer !")
        if question_count == success_count:
            st.balloons()
        button = st.form_submit_button()