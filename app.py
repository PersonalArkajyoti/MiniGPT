import streamlit as st
from openai import AzureOpenAI
from PyPDF2 import PdfReader
import os
import pinecone
import time
import traceback
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AzureOpenAI_Endpoint")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# sidebar_logo = "" #ADD LOGO IF YOU WANT


if not AZURE_OPENAI_KEY or not PINECONE_API_KEY:
    raise ValueError("Azure OpenAI or Pinecone API key not set in environment variables.")

# Initialize Pinecone
def initialize_pinecone():
    try:
        api_key = PINECONE_API_KEY
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set in environment variables.")
        pc = Pinecone(api_key=api_key)
        index_name = PINECONE_INDEX_NAME

        # Delete index if it exists
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            st.info(f"Index '{index_name}' deletion initiated.")
            while index_name in pc.list_indexes():
                time.sleep(1)

        # Create new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        index = pc.Index(index_name)
        st.success("Pinecone initialized and index created.")
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        print("Detailed Error Traceback:")
        print(traceback.format_exc())
        raise

# Process PDF and extract text
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Embed and store in Pinecone
def embed_and_store_pdf(index, pdf_text):
    embed_model = AzureOpenAIEmbeddings(   
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            model=AZURE_EMBED_MODEL,
            deployment=AZURE_DEPLOYMENT
        )  

    chunks = [pdf_text[i: i + 1000] for i in range(0, len(pdf_text), 1000)]
    for i, chunk in enumerate(chunks):
        embedding = embed_model.embed_query(chunk)
        index.upsert([(f"doc_chunk_{i}", embedding, {"text": chunk})])
    st.success("Data embedded and stored successfully.")  

    return len(chunks)

# Query Pinecone
def query_pinecone(index, query_text, embed_model):
    query_embedding = embed_model.embed_query(query_text)
    results = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
    return results

# Main App
def main():
    st.set_page_config(page_title='MiniGPT',page_icon='🧌')
    st.title("🧌MiniGPT - AI Chat & PDF Q&A")
    #st.logo(sidebar_logo)

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION
    )
    embed_model = AzureOpenAIEmbeddings(   
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        model=AZURE_EMBED_MODEL,
        deployment=AZURE_DEPLOYMENT
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
        st.session_state.pdf_text = ""
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "General Chat"
    if "pinecone_index" not in st.session_state:
        st.session_state.pinecone_index = None

    # Sidebar options
    
    
    chat_mode = st.sidebar.radio("Choose chat mode:", ["General Chat", "Chat with PDF"])

    # Clear chat history when mode changes
    if chat_mode != st.session_state.active_mode:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        st.session_state.active_mode = chat_mode

    if chat_mode == "Chat with PDF":
        # Initialize Pinecone only if not already initialized
        if st.session_state.pinecone_index is None:
            st.session_state.pinecone_index = initialize_pinecone()

        # File Upload
        uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file and not st.session_state.pdf_uploaded:
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text:
                st.sidebar.success("PDF processed successfully.")
                st.session_state.pdf_text = pdf_text
                st.session_state.pdf_uploaded = True
                st.session_state.num_chunks = embed_and_store_pdf(st.session_state.pinecone_index, pdf_text)
                st.sidebar.success(f"PDF content split into {st.session_state.num_chunks} chunks and stored in Pinecone.")
            else:
                st.sidebar.error("Unable to process the PDF.")
        elif st.session_state.pdf_uploaded:
            st.sidebar.info("PDF already uploaded and processed.")


    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("Ask me anything!")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if chat_mode == "Chat with PDF" and st.session_state.pdf_uploaded:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing the uploaded document..."):
                
                    results = query_pinecone(st.session_state.pinecone_index, user_input, embed_model)
                    if results.matches:
                        context = "\n".join([match["metadata"]["text"] for match in results.matches])
                        prompt = f"""  
                        You are a smart helpful assistant who gives answers for user queries.
                        Below is context of information:
                        {context}  

                        The user has asked the following question:  
                        "{user_input}"  

                        Provide a clear and structured answer based on the context.
                        """ 
                        chat_prompt = [  
                            {"role": "system", "content": "You are a helpful assistant."},  
                            {"role": "user", "content": prompt},  
                        ]  
                        response = client.chat.completions.create(  
                            model='GEN_AI_AGI_PROCDNA',
                            messages=chat_prompt,  
                            max_tokens=1000,  
                            temperature=0.7,  
                            top_p=0.95,  
                            frequency_penalty=0,  
                            presence_penalty=0,  
                        )  
                        response_text = response.choices[0].message.content
                    else:
                        response_text = "I couldn't find any relevant information in the uploaded document."
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:  # General Chat
             # Append user input to session state
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model='GEN_AI_AGI_PROCDNA',
                        messages=st.session_state.messages,
                        max_tokens=1000,
                        temperature=0.7,  
                        top_p=0.95,  
                        frequency_penalty=0,  
                        presence_penalty=0, 
                    ).choices[0].message.content
                st.markdown(response)  # Display assistant response in chat
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

