import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

st.set_page_config(page_title="Linee Lecco Tech", layout="wide")

# Layout header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("lineelecco_logo.png", width=150)
with col2:
    st.markdown("### **Linee Lecco Tech – Assistenza Tecnica Intelligente**")
    st.markdown("👤 _Tester: Roberto Valentinuzzi_")

st.markdown("---")

st.markdown("📄 Carica uno o più manuali tecnici in PDF, poi inserisci una domanda. L'assistente cercherà la risposta corretta nel contenuto dei documenti.")

uploaded_files = st.file_uploader("**Carica i manuali PDF**", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file caricati. Estrazione in corso...")

    all_text = ""
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()
        for page in pages:
            all_text += page.page_content

    st.info("✅ Estrazione completata. Generazione database vettoriale...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embeddings)

    st.success("🔍 Pronto per le domande!")

    question = st.text_input("❓ Inserisci la tua domanda tecnica")
    if question:
        docs = db.similarity_search(question)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.3, "max_new_tokens": 256}
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)

        st.markdown("### ✅ Risposta trovata:")
        st.success(answer)
else:
    st.warning("📎 Carica almeno un file PDF per iniziare.")
