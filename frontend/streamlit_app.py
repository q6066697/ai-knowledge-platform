import httpx
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Knowledge Platform", layout="wide")
st.title("AI Knowledge Platform")

client = httpx.Client(base_url=API_BASE_URL, timeout=60.0)

st.header("Documents")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
if uploaded_file is not None and st.button("Upload"):
    with st.spinner("Uploading and processing..."):
        try:
            response = client.post(
                "/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            )
        except httpx.RequestError as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "completed":
                    st.success(f"'{data['filename']}' processed: {data['chunks_created']} chunks created.")
                else:
                    st.error(f"Processing failed for '{data['filename']}'.")
            else:
                st.error(f"Upload failed: {response.text}")
    st.rerun()

st.subheader("Uploaded documents")

try:
    docs_response = client.get("/documents")
    documents = docs_response.json() if docs_response.status_code == 200 else []
except httpx.RequestError as exc:
    st.error(f"Could not reach the API: {exc}")
    documents = []

if not documents:
    st.info("No documents uploaded yet.")
else:
    header_cols = st.columns([3, 1, 2, 2, 1])
    for col, label in zip(header_cols, ["Filename", "Type", "Status", "Created at", ""]):
        col.markdown(f"**{label}**")

    for doc in documents:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 1])
        col1.write(doc["filename"])
        col2.write(doc["file_type"])
        col3.write(doc["status"])
        col4.write(doc["created_at"])
        if col5.button("Delete", key=f"delete_{doc['id']}"):
            client.delete(f"/documents/{doc['id']}")
            st.rerun()

st.header("Ask a question")

question = st.text_input("Your question")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            response = client.post("/chat", json={"question": question})
        except httpx.RequestError as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if response.status_code == 200:
                data = response.json()
                st.markdown(f"**Answer:** {data['answer']}")
                if data["sources"]:
                    st.markdown("**Sources:**")
                    for source in data["sources"]:
                        page_info = f", page {source['page']}" if source["page"] is not None else ""
                        st.write(f"- {source['filename']}{page_info} (score: {source['score']:.3f})")
            else:
                st.error(f"Chat request failed: {response.text}")
