# 🧌 MiniGPT - AI Chat & PDF Q&A
## 🔥 Overview
MiniGPT is an AI-powered chatbot that integrates Azure OpenAI for natural language processing and Pinecone for Retrieval-Augmented Generation (RAG) capabilities. Unlike traditional OpenAI-based chatbots, this solution ensures that all processed data remains within Azure’s secure environment, making it an ideal choice for enterprise use cases where data privacy is a concern.

This project also extends beyond basic AI chat functionality by implementing PDF-based Q&A. Users can upload PDFs, and MiniGPT will extract relevant information from them, enabling users to ask document-specific queries.

## 🎯 Problem Statement
Organizations working with sensitive data often need AI-powered assistants but cannot risk exposing their information to public APIs like OpenAI’s ChatGPT. They require a secure, enterprise-grade alternative that:

- Uses Azure OpenAI for data security and compliance.
- Integrates Retrieval-Augmented Generation (RAG) to enhance chatbot responses with document-based knowledge retrieval.
- Allows users to chat with their PDFs, extracting and querying information in real time.
- Provides a seamless UI for both general AI conversations and document-based Q&A.
- MiniGPT addresses these challenges by using Azure OpenAI for secure processing and Pinecone Vector DB for RAG-based document search.

## 🛠 Features
✅ Azure OpenAI-Powered Chatbot – Ensures all AI interactions remain within Azure’s secure environment.
✅ Chat with PDFs – Upload a PDF and query its contents using AI-powered search.
✅ Pinecone Vector Search – Implements RAG to improve AI-generated answers with contextual knowledge from documents.
✅ Multi-Mode Chat – Choose between General Chat and Chat with PDF for AI responses.
✅ FastAPI Backend + Streamlit UI – Easy-to-use interface for quick AI interactions.
✅ Data Privacy & Security – Unlike OpenAI’s ChatGPT, all data remains within Azure’s cloud infrastructure.

## 🚀 Installation & Setup
### 1️⃣ Prerequisites
Ensure you have the following installed:
  - Python 3.11
  - Azure OpenAI Subscription
  - Pinecone API Key
  - Pip & Virtual Environment (Recommended)
### 2️⃣ Clone the Repository
    git clone https://github.com/your-username/MiniGPT-Azure-Pinecone.git
    cd MiniGPT-Azure-Pinecone
### 3️⃣ Create a Virtual Environment (Recommended)
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
  
### 4️⃣ Install Dependencies
  pip install -r requirements.txt
  
### 5️⃣ Set Up Environment Variables
    Create a .env file in the project root and add your API keys:
    
      AZURE_OPENAI_KEY=your_azure_openai_key
      AzureOpenAI_Endpoint=your_azure_endpoint
      AZURE_API_VERSION=your_azure_api_version
      AZURE_OPENAI_MODEL=your_azure_openai_model
      AZURE_EMBED_MODEL=your_azure_embed_model
      AZURE_DEPLOYMENT=your_azure_deployment_name
      PINECONE_API_KEY=your_pinecone_api_key
      PINECONE_INDEX_NAME=your_pinecone_index
      
### 6️⃣ Run the Application
    streamlit run app.py
    
## 🔍 How It Works
### 1️⃣ General Chat Mode
  - The chatbot interacts like ChatGPT, powered by Azure OpenAI.
  - User queries are processed securely within the Azure environment.
### 2️⃣ Chat with PDFs (RAG Implementation)
  - Users upload a PDF document.
  - The PDF is processed, and text is split into vector embeddings.
  - The data is stored in Pinecone Vector DB for efficient search.
  - When users ask a question, the system retrieves relevant document sections and generates context-aware answers.
## 📌 Tech Stack
	
  
  |   Component        |Technology Used           |
  | -------------------| ------------------------ |
  | AI Model           | Azure OpenAI GPT         |
  | Vector Database    | Pinecone                 |
  | Frontend           | Streamlit                |
  | Backend            | FastAPI                  |
  | Document Processing| PyPDF2                   |
  | Embedding Model    | Azure OpenAI text-ada-002|

## 🏆 Why Use MiniGPT?
    🔹 Azure OpenAI Security – No data leaves the Azure environment.
    🔹 RAG-Based Document Q&A – Ask AI questions about uploaded PDFs.
    🔹 Enterprise-Ready – Designed for organizations requiring data privacy.
    🔹 Fast & Scalable – Uses Pinecone for real-time information retrieval.

## 📜 License
This project is open-source and available for modification.

## 👨‍💻 Contributing
Pull requests and suggestions are welcome! Feel free to improve the code or add new features.

## 📞 Contact
For support, reach out via GitHub Issues or email arkajyotichakraborty99@gmail.com
