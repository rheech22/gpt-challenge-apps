import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-4o-mini",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ]
)

questions_prompt = ChatPromptTemplate.from_messages(
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

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)




questions_chain = questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)

formatting_chain = (lambda input: { "context": input.content }) | formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
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
    chain = questions_chain | formatting_chain | output_parser
    return chain.invoke({
        "context": format_docs(_docs),
        "level": level,
    })

@st.cache_data(show_spinner="Making Quiz...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    return retriever.get_relevant_documents(term)

st.title("QuizGPT")

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
            # st.write(docs)
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
    st.write(level)
    quiz = make_quiz(docs, topic if topic else file.name, level)
    st.write(quiz)
    with st.form("questions_form"):
        for q in quiz["questions"]:
            st.write(q["question"])
            value = st.radio(
                "Select an answer.",
                [a["answer"] for a in q["answers"]],
                index=None
            )
            if { "answer": value, "correct": True } in q["answers"]:
                st.success("Correct answer !")
            elif value is not None:
                st.error("Wrong answer !")
        button = st.form_submit_button()