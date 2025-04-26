import os
import streamlit as st
import tempfile
import PyPDF2
from agno.agent import Agent
from agno.models.groq import Groq
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.lancedb import LanceDb
from agno.document.chunking.document import DocumentChunking
from dotenv import load_dotenv
import io


load_dotenv()

def extract_full_pdf_text(uploaded_file_content):
    """Extracts all text from the uploaded PDF file content."""
    try:
        pdf_file = io.BytesIO(uploaded_file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                 text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF for comparison: {e}")
        return None

st.set_page_config(page_title="AI Legal Team Agents", page_icon="⚖️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #3e8e41;'> AI Legal Team Agents </h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 18px; color: #4B0082;'>
        Upload your legal document, optionally provide an updated version for comparison, 
        and let the <b>AI LegalAdvisor</b>, <b>AI ContractsAnalyst</b>, 
        <b>AI LegalStrategist</b>, and <b>AI Team Lead</b> analyze it. 
        The AI can also compare the documents based on summaries.
    </div>
    <br/>
""", unsafe_allow_html=True)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = LanceDb(
        table_name="law", uri="tmp/lancedb", embedder=SentenceTransformerEmbedder()
    )

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "original_doc_text" not in st.session_state:
    st.session_state.original_doc_text = None
if "original_doc_name" not in st.session_state:
    st.session_state.original_doc_name = None
if "updated_doc_text" not in st.session_state:
    st.session_state.updated_doc_text = None
if "updated_doc_name" not in st.session_state:
    st.session_state.updated_doc_name = None

with st.sidebar:
    st.header("⚙️ Configuration")

    chunk_size_in = st.number_input("Chunk Size (for RAG)", min_value=100, max_value=5000, value=1000, step=100)
    overlap_in = st.number_input("Overlap (for RAG)", min_value=0, max_value=1000, value=200, step=50)

    st.divider()

    st.header("1. Upload Original Document")
    uploaded_file = st.file_uploader(
        "Upload the primary Legal Document (PDF)", type=["pdf"], key="original_uploader"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.get("original_doc_name"):
             with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    original_file_content = uploaded_file.getvalue()

                    st.session_state.original_doc_text = extract_full_pdf_text(original_file_content)
                    st.session_state.original_doc_name = uploaded_file.name
                    st.session_state.updated_doc_text = None
                    st.session_state.updated_doc_name = None

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(original_file_content)
                        temp_path = temp_file.name

                    st.session_state.knowledge_base = PDFKnowledgeBase(
                        path=temp_path,
                        vector_db=st.session_state.vector_db,
                        reader=PDFReader(),
                        chunking_strategy=DocumentChunking(chunk_size=chunk_size_in, overlap=overlap_in)
                    )

                    st.session_state.knowledge_base.load(recreate=True, upsert=True)
                    st.session_state.processed_files.add(uploaded_file.name)

                    try:
                       os.unlink(temp_path)
                    except Exception as e:
                       st.warning(f"Could not delete temporary file {temp_path}: {e}")


                    if st.session_state.original_doc_text:
                         st.success(f"Processed '{uploaded_file.name}' for Analysis & Comparison.")
                    else:
                         st.warning(f"Processed '{uploaded_file.name}' for Analysis, but failed to extract full text for comparison feature.")

                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    st.session_state.knowledge_base = None
                    st.session_state.original_doc_text = None
                    st.session_state.original_doc_name = None

    st.divider()

    st.header("2. Upload Updated Document (Optional)")
    if st.session_state.original_doc_text:
        updated_uploaded_file = st.file_uploader(
            "Upload the updated version for comparison", type=["pdf"], key="updated_uploader"
        )

        if updated_uploaded_file:
            if updated_uploaded_file.name != st.session_state.get("updated_doc_name"):
                with st.spinner(f"Loading {updated_uploaded_file.name} for comparison..."):
                    try:
                        updated_file_content = updated_uploaded_file.getvalue()
                        st.session_state.updated_doc_text = extract_full_pdf_text(updated_file_content)
                        st.session_state.updated_doc_name = updated_uploaded_file.name

                        if st.session_state.updated_doc_text:
                            st.success(f"Loaded '{updated_uploaded_file.name}' for comparison.")
                        else:
                             st.error(f"Failed to extract text from '{updated_uploaded_file.name}'. Comparison may not work.")
                    except Exception as e:
                         st.error(f"Error loading updated document: {e}")
                         st.session_state.updated_doc_text = None
                         st.session_state.updated_doc_name = None

    elif st.session_state.original_doc_name and not st.session_state.original_doc_text:
         st.warning("Original document processed for analysis, but text extraction failed. Comparison unavailable.")
    else:
        st.info("Upload an original document first to enable the updated document upload.")


if st.session_state.knowledge_base:
    legal_researcher = Agent(
        name="LegalAdvisor",
        model=Groq(id="gemma2-9b-it"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Legal Researcher AI - Finds and cites relevant legal cases, regulations, and precedents using all data in the knowledge base.",
        instructions=[
            "Extract all available data from the knowledge base and search for legal cases, regulations, and citations related to the user's query.",
            "If needed, use DuckDuckGo for additional legal references AFTER checking the knowledge base.",
            "Always provide source references (e.g., document sections, case names, URLs) in your answers."
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

    contract_analyst = Agent(
        name="ContractAnalyst",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Contract Analyst AI - Reviews contracts and identifies key clauses, risks, and obligations using the full document data.",
        instructions=[
            "Extract all available data from the knowledge base related to the user's query.",
            "Analyze the contract for key clauses (e.g., termination, liability, payment), obligations, potential ambiguities, and risks.",
            "Reference specific sections or clauses from the document where possible."
        ],
        show_tool_calls=True,
        markdown=True
    )

    legal_strategist = Agent(
        name="LegalStrategist",
        model=Groq(id="gemma2-9b-it"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Legal Strategist AI - Provides comprehensive risk assessment and strategic recommendations based on all the available data from the contract.",
        instructions=[
            "Using all data from the knowledge base relevant to the user's query, assess the contract for legal risks and opportunities.",
            "Provide actionable recommendations, suggest alternate clauses if applicable, and ensure compliance with applicable laws based on the provided text.",
            "Clearly explain the reasoning behind recommendations."
        ],
        show_tool_calls=True,
        markdown=True
    )

    team_lead = Agent(
        name="TeamLead",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        description="Team Lead AI - Integrates responses from the Legal Researcher, Contract Analyst, and Legal Strategist into a comprehensive report.",
        instructions=[
            "Combine and synthesize all insights provided by the Legal Researcher, Contract Analyst, and Legal Strategist.",
            "Structure the final output as a coherent legal analysis report.",
            "Ensure the report addresses the user's original query comprehensively.",
            "Include references to relevant sections from the document as provided by the other agents.",
            "Avoid redundancy and present the information clearly and concisely."
        ],
        knowledge=None,
        search_knowledge=False,
        show_tool_calls=True,
        markdown=True
    )

    document_summarizer = Agent(
        name="DocumentSummarizer",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        description="Summarizes legal document text concisely.",
        instructions=[
            "You will be given text from a legal document.",
            "Generate a concise summary focusing on the document's main purpose, key sections/clauses mentioned, core obligations of the parties, and any defined terms or critical definitions.",
            "Keep the summary brief and to the point.",
            "Do not add opinions or analysis, strictly summarize the provided text."
        ],
        knowledge=None,
        search_knowledge=False,
        tools=[],
        show_tool_calls=False,
        markdown=True
    )

    summary_comparator = Agent(
        name="SummaryComparator",
        model=Groq(id="gemma2-9b-it"),
        description="Compares two summaries of different document versions to identify likely key differences between the full documents.",
        instructions=[
             "You are provided with two summaries: 'Summary A' from an original document and 'Summary B' from an updated version.",
             "Carefully compare Summary A and Summary B.",
             "Based *only* on the information present in these summaries, identify and list the likely key differences between the original full documents.",
             "Focus on differences in substance, key terms, obligations, or structure as reflected in the summaries.",
             "Present the likely differences clearly (e.g., using bullet points).",
             "Explicitly state that this comparison is based on summaries and might not capture all detailed textual changes present in the full documents."
        ],
        knowledge=None,
        search_knowledge=False,
        tools=[],
        show_tool_calls=False,
        markdown=True
    )


    def get_team_response(query):
        """Runs the query through the analysis agents and synthesizes the result."""
        st.info("Gathering insights from AI Legal Team...")
        try:
            with st.spinner("Legal Advisor researching..."):
                research_response_obj = legal_researcher.run(query)
                research_response = research_response_obj.content if research_response_obj else "No response from Legal Advisor."
            with st.spinner("Contract Analyst analyzing..."):
                contract_response_obj = contract_analyst.run(query)
                contract_response = contract_response_obj.content if contract_response_obj else "No response from Contract Analyst."
            with st.spinner("Legal Strategist evaluating..."):
                strategy_response_obj = legal_strategist.run(query)
                strategy_response = strategy_response_obj.content if strategy_response_obj else "No response from Legal Strategist."

            st.info("Synthesizing report with AI Team Lead...")
            with st.spinner("Team Lead integrating findings..."):
                final_response_obj = team_lead.run(
                    f"Original Query: {query}\n\n"
                    f"Integrate the following insights gathered using the contract data:\n\n"
                    f"--- Legal Researcher Insights ---\n{research_response}\n\n"
                    f"--- Contract Analyst Insights ---\n{contract_response}\n\n"
                    f"--- Legal Strategist Insights ---\n{strategy_response}\n\n"
                    "Provide a structured legal analysis report addressing the original query, including key terms, obligations, risks, and recommendations, with references to the document sections where available."
                )
            return final_response_obj
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")



if st.session_state.knowledge_base and st.session_state.original_doc_text:

    st.divider()

    st.header("📄 Document Comparison (AI Powered)")
    st.warning("⚠️ This comparison uses AI to summarize each document and then compares the summaries. It may not catch all subtle textual changes and depends on the summary quality.")

    if st.session_state.updated_doc_text:
        st.write(f"Comparing:")
        st.markdown(f"- **Original:** `{st.session_state.original_doc_name}`")
        st.markdown(f"- **Updated:** `{st.session_state.updated_doc_name}`")

        if st.button("Compare Documents using AI Summaries"):
            if st.session_state.original_doc_text.strip() == st.session_state.updated_doc_text.strip():
                 st.success("✅ The document texts appear to be identical.")
            else:
                summary_a = None
                summary_b = None
                comparison_result_content = None

                with st.spinner("Step 1/3: Summarizing Original Document with AI Agent..."):
                    try:
                        if not st.session_state.original_doc_text.strip():
                             raise ValueError("Original document text is empty or could not be read.")

                        if 'document_summarizer' not in locals():
                             raise NameError("DocumentSummarizer agent is not defined.")

                        max_summary_input = 7500
                        prompt_a = f"Summarize the following legal document text:\n\n```\n{st.session_state.original_doc_text[:max_summary_input]}\n```"
                        response_a = document_summarizer.run(prompt_a)
                        if response_a and response_a.content:
                            summary_a = response_a.content
                            st.info("Original document summarized.")
                        else:
                             raise Exception("Summarizer agent did not return content for Original Document.")
                    except Exception as e:
                        st.error(f"Error summarizing original document: {e}")

                if summary_a:
                    with st.spinner("Step 2/3: Summarizing Updated Document with AI Agent..."):
                         try:
                            if not st.session_state.updated_doc_text.strip():
                                 raise ValueError("Updated document text is empty or could not be read.")

                            if 'document_summarizer' not in locals():
                                raise NameError("DocumentSummarizer agent is not defined.")

                            prompt_b = f"Summarize the following legal document text:\n\n```\n{st.session_state.updated_doc_text[:max_summary_input]}\n```"
                            response_b = document_summarizer.run(prompt_b)
                            if response_b and response_b.content:
                                summary_b = response_b.content
                                st.info("Updated document summarized.")
                            else:
                                 raise Exception("Summarizer agent did not return content for Updated Document.")
                         except Exception as e:
                             st.error(f"Error summarizing updated document: {e}")

                if summary_a and summary_b:
                    with st.spinner("Step 3/3: Comparing Summaries with AI Agent..."):
                        try:
                            if 'summary_comparator' not in locals():
                                raise NameError("SummaryComparator agent is not defined.")

                            comparison_prompt = (
                                f"Summary A (Original: {st.session_state.original_doc_name}):\n{summary_a}\n\n"
                                f"Summary B (Updated: {st.session_state.updated_doc_name}):\n{summary_b}\n\n"
                                f"Based ONLY on the two summaries above, what are the likely key differences between the original and updated full documents? State clearly this is summary-based."
                            )
                            comparison_response = summary_comparator.run(comparison_prompt)
                            if comparison_response and comparison_response.content:
                                comparison_result_content = comparison_response.content
                                st.info("Comparison complete.")
                            else:
                                raise Exception("Comparison agent did not return content.")
                        except Exception as e:
                             st.error(f"Error comparing summaries: {e}")

                st.markdown("### AI Comparison Result (Based on Summaries):")
                if comparison_result_content:
                    st.markdown(comparison_result_content)
                else:
                    st.warning("Could not generate an AI comparison based on summaries. Check summaries below.")

                with st.expander("View Generated Summaries Used for Comparison"):
                    st.markdown("#### Summary of Original:")
                    st.markdown(summary_a if summary_a else "_Summary failed or not generated._")
                    st.markdown("---")
                    st.markdown("#### Summary of Updated:")
                    st.markdown(summary_b if summary_b else "_Summary failed or not generated._")

    elif st.session_state.original_doc_name and st.session_state.original_doc_text:
         st.info("Upload an 'Updated Document' in the sidebar to enable comparison.")
    elif st.session_state.original_doc_name and not st.session_state.original_doc_text:
         st.warning("Original document processed for analysis, but text extraction failed. Comparison unavailable.")

    st.divider()

    st.header("🔍 Document Analysis")
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"]
    )

    query = None
    if analysis_type == "Custom Query":
        query = st.text_area("Enter your custom legal question:", key="custom_query_input")
    else:
        predefined_queries = {
            "Contract Review": (
                "Analyze this legal document from the knowledge base. "
                "Identify and detail key terms, clauses (like termination, liability, payment, confidentiality), parties' obligations, and potential risks or ambiguities."
            ),
            "Legal Research": (
                "Based on the content of the document in the knowledge base, find relevant legal cases, statutes, or precedents. "
                "Focus on aspects mentioned in the document (e.g., specific clauses, jurisdiction if mentioned). Provide detailed references and sources if possible."
            ),
            "Risk Assessment": (
                "Extract data from the knowledge base for this document. Identify potential legal and commercial risks for the parties involved. "
                "Detail specific risk areas (e.g., liability exposure, termination conditions, IP rights) and suggest potential mitigation strategies or alternative clauses if appropriate."
            ),
            "Compliance Check": (
                "Evaluate the document in the knowledge base for compliance with common legal regulations or standards relevant to its subject matter (e.g., data privacy if applicable, standard contract terms). "
                "Highlight any areas of potential non-compliance or concern based on the text, and suggest corrective actions or clauses if possible."
            )
        }
        query = predefined_queries[analysis_type]
        st.markdown(f"**Query:** _{query}_")

    if st.button("Analyze Document"):
        if not query:
            st.warning("Please enter a custom query or select a predefined analysis type.")
        else:
            with st.spinner("AI Legal Team is analyzing... This may take a moment."):
                response_obj = get_team_response(query)

                if response_obj and response_obj.content:
                     tabs = st.tabs(["📝 Full Analysis Report", "📌 Key Points", "📋 Recommendations"])

                     with tabs[0]:
                         st.subheader("📑 Detailed Analysis Report")
                         st.markdown(response_obj.content)

                     with tabs[1]:
                         st.subheader("📌 Key Points Summary")
                         with st.spinner("Generating Key Points..."):
                             key_points_response = team_lead.run(
                                 f"Based on the following full analysis report, extract and list the most critical key points (e.g., main obligations, major risks, key definitions) in a concise bulleted list:\n\n{response_obj.content}"
                             )
                             st.markdown(key_points_response.content if key_points_response and key_points_response.content else "Could not generate key points summary.")

                     with tabs[2]:
                         st.subheader("📋 Recommendations")
                         with st.spinner("Generating Recommendations..."):
                             recommendations_response = team_lead.run(
                                 f"Based on the following full analysis report, extract only the specific, actionable legal recommendations provided. List them clearly:\n\n{response_obj.content}"
                             )
                             st.markdown(recommendations_response.content if recommendations_response and recommendations_response.content else "Could not generate recommendations summary.")
                else:
                     st.error("Analysis failed or returned no content. Please check the document or try a different query.")

elif not st.session_state.original_doc_name:
     st.info("👈 Please upload a legal document using the sidebar to begin analysis and comparison.")