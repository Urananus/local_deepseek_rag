from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
import gradio as gr
import asyncio
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings, OllamaLLM

async def async_initialize_rag(pdf_files):
    try:
        # Process PDF files and create documents
        documents = []
        for pdf_file in pdf_files:
            with open(pdf_file.name, "rb") as f:
                pdf = PdfReader(f)
                documents.extend([
                    Document(
                        page_content=page.extract_text(),
                        metadata={"source": pdf_file.name, "page": i+1}
                    )
                    for i, page in enumerate(pdf.pages)
                    if page.extract_text().strip()  # Skip empty pages
                ])

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(
            model="deepseek-r1"
        
        )
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Create QA pipeline
        llm = OllamaLLM(
            model="deepseek-r1",
            temperature=0.3,
            num_ctx=4096
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

async def async_ask_question(query, pipeline):
    try:
        response = await asyncio.to_thread(pipeline.invoke, {"query": query})
        answer = response["result"]
        sources = "\n".join(
            f"• {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})"
            for doc in response["source_documents"]
        )
        return f"{answer}\n\nSources:\n{sources}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

def create_interface():
    with gr.Blocks(title="Advanced RAG System") as interface:
        gr.Markdown("# PDF Knowledge Assistant\nUpload PDFs and ask questions")
        
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    file_count="multiple",
                    label="Upload PDF Documents",
                    type="filepath"
                )
                init_btn = gr.Button("Initialize System", variant="primary")
                status = gr.Textbox(label="System Status", interactive=False)
            
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the documents...",
                    lines=3
                )
                ask_btn = gr.Button("Ask Question", variant="secondary")
                response_output = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=6
                )
        
        # State management
        pipeline_state = gr.State()

        # Event handlers
        init_btn.click(
            fn=async_initialize_rag,
            inputs=file_upload,
            outputs=pipeline_state,
            api_name="init_rag"
        ).then(
            lambda: "System Ready - You can now ask questions",
            outputs=status
        )

        ask_btn.click(
            fn=async_ask_question,
            inputs=[question_input, pipeline_state],
            outputs=response_output,
            api_name="ask_question"
        )

    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )