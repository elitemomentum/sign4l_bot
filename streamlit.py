import streamlit as st
import os
import zipfile
import tempfile
import time
from pinecone import Pinecone, ServerlessSpec

# ========== CONFIG ==========

API_KEY = "pcsk_w9Hxy_5x4QjXh7o17b2iGpdPGTGUDYiMbD6KAgJitQtucLUVL7tk9ckkDDSevoFZNECjq"  # Replace with your actual Pinecone API key
INDEX_NAME = "project-summary-assistant"
REGION = "us-west1"  # Adjust as needed
NAMESPACE = "pdf-docs"

# ========== INIT ==========

pc = Pinecone(api_key=API_KEY)

def create_index():
    """Create a Pinecone index for document storage"""
    try:
        # Check if index already exists
        if INDEX_NAME in [index.name for index in pc.list_indexes()]:
            return f"[✓] Index '{INDEX_NAME}' already exists."
        
        # Create new serverless index
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Standard for text embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=REGION
            )
        )
        return f"[✓] Index '{INDEX_NAME}' created successfully."
    except Exception as e:
        return f"[!] Failed to create index: {e}"

def upload_zip_and_docs(zip_file):
    """Process a ZIP file and upload PDF documents to the index"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        pdf_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".pdf")]
        upload_status = []
        
        # Get the index
        index = pc.Index(INDEX_NAME)
        
        # Process each PDF file
        for pdf in pdf_files:
            file_path = os.path.join(tmpdir, pdf)
            try:
                # Here you would extract text and create embeddings
                # For demonstration, we'll use a placeholder approach
                # In a real app, you'd use a text extraction library and embedding model
                
                # Simple placeholder approach (not for production)
                with open(file_path, "rb") as f:
                    # Just read first 1000 bytes to simulate processing
                    content = f.read(1000)
                    
                # Create a dummy vector (in production, use proper embeddings)
                dummy_vector = [0.0] * 1536  # Match your dimension
                
                # Create a unique ID for this document
                doc_id = f"doc_{pdf.replace('.', '_')}"
                
                # Upsert to index
                index.upsert(
                    vectors=[
                        {
                            "id": doc_id,
                            "values": dummy_vector,
                            "metadata": {
                                "filename": pdf,
                                "source": "streamlit_upload"
                            }
                        }
                    ],
                    namespace=NAMESPACE
                )
                
                upload_status.append(f"[✓] Processed and indexed: {pdf}")
            except Exception as e:
                upload_status.append(f"[!] Failed to process {pdf}: {e}")
        
        # Store the upload time for tracking processing
        st.session_state['upload_time'] = time.time()
        return upload_status

def check_index_status():
    """Check if index is ready for querying"""
    if 'upload_time' not in st.session_state:
        return False, "No files have been uploaded yet."
    
    # Wait at least 30 seconds after upload before querying
    if time.time() - st.session_state.get('upload_time', 0) < 30:
        remaining = 30 - int(time.time() - st.session_state.get('upload_time', 0))
        return False, f"Index still processing. Please wait about {remaining} seconds before querying."
    
    try:
        # Check if index exists
        if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            return False, f"Index '{INDEX_NAME}' does not exist."
        
        return True, "Index should be ready for querying."
    except Exception as e:
        return False, f"Error checking index status: {e}"

def ask_question(query):
    """Process a user query and return a response"""
    try:
        # In a real implementation, you would:
        # 1. Convert the query to an embedding
        # 2. Search the Pinecone index for relevant documents
        # 3. Use retrieved contexts to generate a response
        
        # Get the index
        index = pc.Index(INDEX_NAME)
        
        # Create a dummy query vector (use real embeddings in production)
        query_vector = [0.0] * 1536
        
        # Query the index
        query_response = index.query(
            namespace=NAMESPACE,
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        # Extract matched documents
        matched_docs = []
        for match in query_response.matches:
            if hasattr(match, 'metadata') and match.metadata:
                matched_docs.append(f"- {match.metadata.get('filename', 'Unknown document')}")
        
        # Generate a response (in a real app, you'd use an LLM here)
        if matched_docs:
            response = f"Based on the documents I found:\n\n"
            response += "\n".join(matched_docs)
            response += f"\n\nRegarding your query: '{query}'\n\n"
            response += "I would need to analyze the document content with an LLM to provide specific answers."
        else:
            response = f"I couldn't find relevant documents to answer your query: '{query}'"
        
        return response
    except Exception as e:
        return f"Error processing your question: {e}"

def delete_index():
    """Delete the Pinecone index"""
    try:
        # Check if index exists before attempting deletion
        if INDEX_NAME in [index.name for index in pc.list_indexes()]:
            pc.delete_index(INDEX_NAME)
            # Clear session state when index is deleted
            if 'upload_time' in st.session_state:
                del st.session_state['upload_time']
            return f"[✓] Index '{INDEX_NAME}' deleted successfully."
        else:
            return f"[!] Index '{INDEX_NAME}' does not exist."
    except Exception as e:
        return f"[!] Failed to delete index: {e}"

# ========== UI ==========

st.title("📄 Pinecone PDF Assistant")
st.markdown("Upload a ZIP of PDFs, ask questions, or delete your assistant.")

# Initialize session state for tracking file uploads
if 'upload_time' not in st.session_state:
    st.session_state['upload_time'] = 0

tab1, tab2, tab3 = st.tabs(["➕ Upload ZIP", "❓ Ask Question", "🗑️ Delete Index"])

with tab1:
    st.header("Upload PDFs")
    uploaded_file = st.file_uploader("Upload a ZIP file of PDFs", type="zip")

    if st.button("Create Index"):
        with st.spinner("Creating index..."):
            result = create_index()
            st.success(result)

    if uploaded_file and st.button("Upload and Index PDFs"):
        with st.spinner("Uploading and indexing PDFs..."):
            try:
                status = upload_zip_and_docs(uploaded_file)
                st.success("All files processed! Please wait ~30 seconds before querying.")
                for line in status:
                    st.text(line)
            except Exception as e:
                st.error(f"[!] Upload failed: {e}")

with tab2:
    st.header("Ask a Question")
    
    # Display index status
    index_ready, status_msg = check_index_status()
    if index_ready:
        st.success(status_msg)
    else:
        st.warning(status_msg)
    
    user_query = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if user_query:
            index_ready, status_msg = check_index_status()
            if not index_ready:
                st.warning(status_msg)
            else:
                with st.spinner("Getting answer..."):
                    try:
                        answer = ask_question(user_query)
                        st.success("Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Try deleting the index and starting over, or wait longer for processing.")
        else:
            st.warning("Please enter a question.")

with tab3:
    st.header("Delete Index")
    if st.button("Delete Index"):
        with st.spinner("Deleting index..."):
            result = delete_index()
            st.warning(result)

# Display note about implementation
st.sidebar.markdown("""
## Note

This is a simplified implementation that demonstrates the workflow. 
For a production application, you would need to:

1. Use a PDF text extraction library
2. Generate proper text embeddings 
3. Integrate with an LLM API for answering questions

The current implementation uses placeholder vectors and simulated responses.
""")