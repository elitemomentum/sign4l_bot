import streamlit as st
import os
import zipfile
import tempfile
import time
from pinecone import Pinecone

# ========== CONFIG ==========

API_KEY = "pcsk_w9Hxy_5x4QjXh7o17b2iGpdPGTGUDYiMbD6KAgJitQtucLUVL7tk9ckkDDSevoFZNECjq"  # Replace with your actual Pinecone API key
ASSISTANT_NAME = "project-summary-assistant"
REGION = "us"

# ========== INIT ==========

pc = Pinecone(api_key=API_KEY)

def create_assistant():
    try:
        assistant = pc.assistant.create_assistant(
            assistant_name=ASSISTANT_NAME,
            instructions="Answer based only on the documents provided. Use clear American English.",
            region=REGION,
            timeout=30
        )
        return f"[✓] Assistant '{ASSISTANT_NAME}' created."
    except Exception as e:
        return f"[!] Assistant might already exist or failed to create. {e}"

def upload_zip_and_docs(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
        pdf_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".pdf")]
        upload_status = []
        file_ids = []

        for pdf in pdf_files:
            file_path = os.path.join(tmpdir, pdf)
            response = assistant.upload_file(
                file_path=file_path,
                metadata={"source": "streamlit_upload"},
                timeout=None
            )
            # Store file ID if available in response
            if response and hasattr(response, 'get') and response.get('id'):
                file_ids.append(response.get('id'))
            upload_status.append(f"[✓] Uploaded: {pdf}")
            
        # Store file IDs in session state for verification later
        st.session_state['file_ids'] = file_ids
        st.session_state['upload_time'] = time.time()
        return upload_status

def check_files_status():
    """Check if files are processed and ready for querying"""
    if 'file_ids' not in st.session_state:
        return False, "No files have been uploaded yet."
    
    # Wait at least 30 seconds after upload before querying
    if time.time() - st.session_state.get('upload_time', 0) < 30:
        remaining = 30 - int(time.time() - st.session_state.get('upload_time', 0))
        return False, f"Files still processing. Please wait about {remaining} seconds before querying."
    
    try:
        # Optional: You can add an API call here to check file status if Pinecone provides one
        return True, "Files should be ready for querying."
    except Exception as e:
        return False, f"Error checking file status: {e}"

def ask_question(query):
    assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
    # Use dictionary format for messages instead of Message class
    msg = {"role": "user", "content": query}
    resp = assistant.chat(messages=[msg])
    return resp['message']['content']

def delete_assistant():
    try:
        pc.assistant.delete_assistant(assistant_name=ASSISTANT_NAME)
        # Clear session state when assistant is deleted
        if 'file_ids' in st.session_state:
            del st.session_state['file_ids']
        if 'upload_time' in st.session_state:
            del st.session_state['upload_time']
        return f"[✓] Assistant '{ASSISTANT_NAME}' deleted successfully."
    except Exception as e:
        return f"[!] Failed to delete assistant: {e}"

# ========== UI ==========

st.title("📄 Pinecone PDF Assistant")
st.markdown("Upload a ZIP of PDFs, ask questions, or delete your assistant.")

# Initialize session state for tracking file uploads
if 'file_ids' not in st.session_state:
    st.session_state['file_ids'] = []
if 'upload_time' not in st.session_state:
    st.session_state['upload_time'] = 0

tab1, tab2, tab3 = st.tabs(["➕ Upload ZIP", "❓ Ask Question", "🗑️ Delete Assistant"])

with tab1:
    st.header("Upload PDFs")
    uploaded_file = st.file_uploader("Upload a ZIP file of PDFs", type="zip")

    if st.button("Create Assistant"):
        with st.spinner("Creating assistant..."):
            result = create_assistant()
            st.success(result)

    if uploaded_file and st.button("Upload and Index PDFs"):
        with st.spinner("Uploading and indexing PDFs..."):
            try:
                status = upload_zip_and_docs(uploaded_file)
                st.success("All files uploaded successfully! Please wait ~30 seconds before querying.")
                for line in status:
                    st.text(line)
            except Exception as e:
                st.error(f"[!] Upload failed: {e}")

with tab2:
    st.header("Ask your Assistant")
    
    # Display file status
    files_ready, status_msg = check_files_status()
    if files_ready:
        st.success(status_msg)
    else:
        st.warning(status_msg)
    
    user_query = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if user_query:
            files_ready, status_msg = check_files_status()
            if not files_ready:
                st.warning(status_msg)
            else:
                with st.spinner("Getting answer..."):
                    try:
                        answer = ask_question(user_query)
                        st.success("Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Try deleting the assistant and starting over, or wait longer for file processing.")
        else:
            st.warning("Please enter a question.")

with tab3:
    st.header("Delete Assistant")
    if st.button("Delete Assistant"):
        with st.spinner("Deleting assistant..."):
            result = delete_assistant()
            st.warning(result)