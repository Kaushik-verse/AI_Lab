# ------------------ app.py ------------------
import streamlit as st  # Streamlit library for building web apps
import main as main  # Import functions from main.py

# Set the title of the app
st.title("Chat with PDFs with Deepseek")

# File uploader for PDFs
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

# If a file is uploaded
if uploaded_file:
    # Save PDF to local folder
    main.upload_pdf(uploaded_file)

    # Create vector store (embeddings) for the uploaded PDF
    db = main.create_vector_store(main.pdfs_directory + uploaded_file.name)

    # Chat input for user to type a question
    question = st.chat_input()

    # If the user submits a question
    if question:
        # Display user's message in chat
        st.chat_message("user").write(question)

        # Retrieve top relevant documents from the PDF
        related_documents = main.retrieve_docs(db, question)

        # Ask the AI model the question using the retrieved documents as context
        answer = main.question_pdf(question, related_documents)

        # Display AI's answer in chat
        st.chat_message("assistant").write(answer)