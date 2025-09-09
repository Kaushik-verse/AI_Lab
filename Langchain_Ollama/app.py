import streamlit as st
import os
import main as main

st.title("📄 Chat with PDFs using Local LLM (Ollama + LangChain)")

# Initialize session state
if "context_history" not in st.session_state:
    st.session_state.context_history = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Multi-PDF uploader
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Save PDFs locally
    for file in uploaded_files:
        main.upload_pdf(file)

    # Build vector store from uploaded PDFs
    file_paths = [os.path.join(main.pdfs_directory, f.name) for f in uploaded_files]
    db, all_docs = main.create_vector_store(file_paths)

    # Tabs: Chat / Summarize / Export
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Summarize PDFs", "📁 Export Data"])

    with tab1:
        question = st.chat_input("Ask a question about the PDFs:")
        if question:
            st.chat_message("user").write(question)
            related_docs = main.retrieve_docs(db, question)
            context_with_history = st.session_state.context_history + "\n\n" + question
            answer = main.question_pdf(question, related_docs, context_with_history)
            st.chat_message("assistant").write(answer)

            # Update session state
            st.session_state.context_history = context_with_history + "\n\n" + answer
            st.session_state.chat_history.append({"question": question, "answer": answer})

    with tab2:
        if st.button("Generate Summary"):
            summary = main.summarize_pdf(all_docs)
            st.write("**Summary:**")
            st.write(summary)
            st.session_state.summary = summary

    with tab3:
        if st.button("Export Chat History as CSV"):
            csv_data = "Question,Answer\n"
            for item in st.session_state.chat_history:
                csv_data += f"\"{item['question']}\",\"{item['answer']}\"\n"
            st.download_button(
                label="Download Chat History",
                data=csv_data,
                file_name="chat_history.csv",
                mime="text/csv"
            )

        if st.button("Export Summary as TXT"):
            summary_text = st.session_state.get("summary", "No summary generated yet.")
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name="pdf_summary.txt",
                mime="text/plain"
            )