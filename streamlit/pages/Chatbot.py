# Import python packages
import pandas as pd
import streamlit as st
from snowflake.core import Root
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.core import Root
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.cortex import Complete, ExtractAnswer, Sentiment, Summarize, Translate, ClassifyText

from snowflake.snowpark.context import get_active_session

# Import python packages
import streamlit as st
from snowflake.core import Root
from snowflake.snowpark.context import get_active_session

# Constants
DB = "productionfloor_db"
SCHEMA = "public"
SERVICE = "cortex_search_production_data"
BASE_TABLE = "SENSORS_AD_DETECTED"

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title(":gear: Manufacturing Floor Q&A Assistant  :gear:")
st.caption(
    f"""Welcome! This application suggests answers to questions based 
    on the available data and previous agent responses in support chats.
    """
)

# Get current credentials
session = get_active_session()

# Constants
CHAT_MEMORY = 10


MODELS = [
    "mistral-large",
    "llama3.1-70b",
    "llama3.1-8b",
    "mistral-large2",
    "llama3.1-405b",
]

def init_messages():
    """
    Initialize the session state for chat messages. If the session state indicates that the
    conversation should be cleared or if the "messages" key is not in the session state,
    initialize it as an empty list.
    """
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None



##########################################
#       Cortex Search
##########################################
def init_service_metadata():
    """
    Initialize the session state for cortex search service metadata. 
    """
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append(
                    {"name": svc_name, "search_column": svc_search_col}
                )

        st.session_state.service_metadata = service_metadata

def init_config_options():
    """
    Initialize the configuration options in the Streamlit sidebar. Allow the user to select
    a cortex search service, clear the conversation, toggle debug mode, and toggle the use of
    chat history. Also provide advanced options to select a model, the number of context chunks,
    and the number of chat messages to use in the chat history.
    """
    st.sidebar.selectbox(
        "Select cortex search service:",
        [s["name"] for s in st.session_state.service_metadata],
        key="selected_cortex_search_service",
    )

    st.sidebar.button("Clear conversation", key="clear_conversation")
    st.sidebar.toggle("Debug", key="debug", value=False)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input(
            "Select number of context chunks",
            value=5,
            key="num_retrieved_chunks",
            min_value=1,
            max_value=10,
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=5,
            key="num_chat_messages",
            min_value=1,
            max_value=10,
        )

    st.sidebar.expander("Session State").write(st.session_state)

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def query_cortex_search_service(query):
    """
    Query the selected cortex search service with the given query and retrieve context data.

    Args:
        query (str): The query to search the cortex search service with.

    Returns:
        str: The concatenated string of context.
    """
    db, schema = session.get_current_database(), session.get_current_schema()

    cortex_search_service = (
        root.databases[db]
        .schemas[schema]
        .cortex_search_services[st.session_state.selected_cortex_search_service]
    )

    context_documents = cortex_search_service.search(
        query, columns=[], limit=st.session_state.num_retrieved_chunks
    )
    results = context_documents.results

    service_metadata = st.session_state.service_metadata
    search_col = [s["search_column"] for s in service_metadata
                    if s["name"] == st.session_state.selected_cortex_search_service][0]

    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context {i+1}: {r[search_col]} \n" + "\n"

    if st.session_state.debug:
        st.sidebar.text_area("Context", context_str, height=500)

    return context_str


def get_chat_history():
    """
    Retrieve the chat history from the session state limited to the number of messages specified
    by the user in the sidebar options.

    Returns:
        list: The list of chat messages from the session state.
    """
    start_index = max(
        0, len(st.session_state.messages) - st.session_state.num_chat_messages
    )
    return st.session_state.messages[start_index : len(st.session_state.messages) - 1]

def complete(model, prompt):
    """
    Generate a completion for the given prompt using the specified model.

    Args:
        model (str): The name of the model to use for completion.
        prompt (str): The prompt to generate a completion for.

    Returns:
        str: The generated completion.
    """
    return session.sql("SELECT snowflake.cortex.complete(?,?)", (model, prompt)).collect()[0][0]

def make_chat_history_summary(chat_history, question):
    """
    Generate a summary of the chat history combined with the current question to extend the query
    context. Use the language model to generate this summary.

    Args:
        chat_history (str): The chat history to include in the summary.
        question (str): The current user question to extend with the chat history.

    Returns:
        str: The generated summary of the chat history and question.
    """
    
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """

    summary = complete(st.session_state.model_name, prompt)

    if st.session_state.debug:
        st.sidebar.text_area(
            "Chat history summary", summary.replace("$", "\$"), height=150
        )

    return summary

def create_prompt(user_question):
    """
    Create a prompt for the language model by combining the user question with context retrieved
    from the cortex search service and chat history (if enabled). Format the prompt according to
    the expected input format of the model.

    Args:
        user_question (str): The user's question to generate a prompt for.

    Returns:
        str: The generated prompt for the language model.
    """
    
    if st.session_state.use_chat_history:
        
        chat_history = get_chat_history()
        if chat_history != []:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
        else:
            
            prompt_context = query_cortex_search_service(user_question)
            question_summary=''
                
        

    prompt = f"""
            You are a AI assistant.There is data from various sensors monitoring a machine. Each sensor records different parameters such as vibration levels, temperature, motor_amps, motor_rpm sensors. The goal is to analyze this data for potential anomalies and gain insights into the machine's performance.
            Answer this question by extracting information given between
           between <context> and </context> tags. \n
            When presented with the question use the information between the \n
            <context> and </context> tags.You are offering a chat experience using the 
            user's chat history provided in between the <chat_history> and </chat_history> tags
            to provide a summary that addresses the user's question. 
        
           When answering the question be concise and dont provide explanation.           
           If you don´t have the information just say so.

           
           The question is given between the <question> and </question> tags.

           <chat_history>
            {chat_history}
            </chat_history>
            <context>
            {prompt_context}
            </context>
            
           <question>
           {user_question}
           </question>
     
           Answer:
        """
    prompt = prompt.replace("'", "''")
    return prompt

##########################################
#      Main
##########################################

def main():
    st.title(f":speech_balloon: Chatbot with Snowflake Cortex")

    init_service_metadata()
    init_config_options()
    init_messages()

    icons = {"assistant": "❄️", "user": "👤"}
    
        
    disable_chat = (
        "service_metadata" not in st.session_state
        or len(st.session_state.service_metadata) == 0
    )
    if question := st.chat_input("Ask a question...", disabled=disable_chat):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user", avatar=icons["user"]):
            st.markdown(question.replace("$", "\$"))
            
        

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=icons["assistant"]):
            message_placeholder = st.empty()
            # question = question.replace("'", "")
            with st.spinner("Thinking..."):
                # Generate the response
                generated_response = complete(
                    st.session_state.model_name, create_prompt(question)
                )
                
                # Store the generated response directly in session state
                st.session_state.gen_response = generated_response
                
                # Display the generated response
                message_placeholder.markdown(generated_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": generated_response}
        )
        

if __name__ == "__main__":
    session = get_active_session()
    root = Root(session)
    main()