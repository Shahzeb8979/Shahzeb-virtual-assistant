import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# LangChain components for Retrieval & Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Native Hugging Face API Client
from huggingface_hub import InferenceClient

# 1. Load environment variables
load_dotenv(override=True)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("⚠️ `HUGGINGFACEHUB_API_TOKEN` missing! Please check your `.env` file.")
    st.stop()

# 2. Streamlit Page Configuration
st.set_page_config(
    page_title="Shahzeb Adil | AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Dynamic Experience & Age Calculator Helper
def calculate_live_metrics():
    """Calculates total experience dynamically based on current date."""
    now = datetime.now()
    career_start = datetime(2019, 11, 1)
    
    total_months = (now.year - career_start.year) * 12 + (now.month - career_start.month)
    exp_years = total_months // 12
    exp_rem_months = total_months % 12
    
    exp_string = f"{exp_years} yrs {exp_rem_months} mos" if exp_rem_months > 0 else f"{exp_years} yrs"
    approx_age = now.year - 1997
    
    return exp_string, approx_age

TOTAL_EXPERIENCE, CURRENT_AGE = calculate_live_metrics()

# 4. Ultra-Clean & Premium CSS Injection
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global App Canvas */
    .stApp {
        background-color: #080c14;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1050px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1321 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Sidebar Profile Header */
    .sidebar-profile {
        text-align: center;
        padding: 1.25rem 1rem;
        background: linear-gradient(180deg, #131c2e 0%, #0d1321 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        margin-bottom: 1.25rem;
    }

    .sidebar-avatar {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        margin: 0 auto 0.75rem auto;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
    }

    /* Info Badge Container */
    .info-card {
        background: #131c2e;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.8;
    }

    .info-card strong {
        color: #f1f5f9;
    }

    /* Hero Header Styling */
    .hero-banner {
        padding: 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(19, 28, 46, 0.9) 0%, rgba(13, 19, 33, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.2) 0%, rgba(0, 0, 0, 0) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }

    .hero-title span {
        background: linear-gradient(135deg, #818cf8 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        font-size: 0.98rem;
        color: #94a3b8;
        max-width: 650px;
        line-height: 1.6;
    }

    .pulse-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1.2rem;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }

    /* Metric Cards Styling Override */
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #818cf8 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #94a3b8 !important;
    }

    /* Quick Prompt Buttons */
    .stButton > button {
        background: #131c2e !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        background: #1e293b !important;
        border-color: #6366f1 !important;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }

    /* Chat Messages & Input Container Fixes */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0.8rem 0 !important;
    }

    /* Chat Input Container Outer Box */
    .stChatInputContainer, 
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        background-color: #0d1321 !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
    }

    .stChatInputContainer:focus-within,
    [data-testid="stChatInput"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4) !important;
    }

    /* Chat Textarea Typing Color Fix */
    .stChatInputTextArea, 
    .stChatInputContainer textarea,
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background-color: #0d1321 !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
    }

    .stChatInputContainer textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #080c14; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# 5. Sidebar Profile & Actions
with st.sidebar:
    st.markdown("""
        <div class="sidebar-profile">
            <div class="sidebar-avatar">🤖</div>
            <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Shahzeb's AI Assistant</h3>
            <p style="margin: 4px 0 0 0; color: #818cf8; font-size: 0.8rem; font-weight: 600;">GenAI & Agentic AI Engineer</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="info-card">
            💼 <strong>Total Experience:</strong> {TOTAL_EXPERIENCE}<br>
            📍 <strong>Location:</strong> Bareilly, UP, India<br>
            ✉️ <strong>Email:</strong> adil.shahzeb9499@gmail.com<br>
            📱 <strong>Phone:</strong> +91 73009 42122
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 0.8rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <p style="margin: 0; font-weight: 600; color: #f8fafc; font-size: 0.85rem;">🎓 Education & Credentials</p>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.8rem;">
                <strong>AlmaBetter x IIT Patna</strong><br>
                Fellowship in Data Science & AI<br><br>
                <strong>Dr. A.P.J. AKTU</strong><br>
                B.Tech in Computer Science
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Resume Download Button
    try:
        with open("Shahzeb_Adil_Resume.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Download PDF Resume",
                data=pdf_file,
                file_name="Shahzeb_Adil_Resume.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except FileNotFoundError:
        pass

# 6. Hero Banner Header & Dynamic Metrics
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Shahzeb's <span>Virtual Assistant</span></div>
        <div class="hero-sub">Get instant, dynamic insights into Shahzeb's work experience, GenAI & RAG architectures, technical stack, and career background.</div>
        <div class="pulse-badge">
            <div class="pulse-dot"></div> System Active & RAG Engine Connected
        </div>
    </div>
""", unsafe_allow_html=True)

# Dynamic Stat Counter Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Total Experience", value=TOTAL_EXPERIENCE)
with m2:
    st.metric(label="Core Specialty", value="GenAI & RAG")
with m3:
    st.metric(label="Vector Store", value="FAISS DB")
with m4:
    st.metric(label="LLM Model", value="Qwen 2.5 7B")

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# 7. RAG Engine & Embedding Setup
@st.cache_resource
def setup_vector_store():
    resume_content = f"""
    SHAHZEB ADIL
    GENAI ENGINEER | RAG | PROMPT ENGINEERING | AI AUTOMATION
    Location: Bareilly, UP, 243003
    Phone: +91 73009 42122
    Email: adil.shahzeb9499@gmail.com

    CORE FACTS & METRICS:
    - Total Work Experience: {TOTAL_EXPERIENCE} (Career started in Nov 2019 across Software Dev, AI/ML, and Tech Support).
    - Current Age: Approximately {CURRENT_AGE} years old.

    PROFESSIONAL SUMMARY:
    Innovative Generative AI Developer with expertise in LLM applications, AI automation, and prompt engineering. Proven track record in building scalable AI solutions using Python, LangChain, Hugging Face, and vector databases.

    TECHNICAL SKILLS:
    - Generative & Agentic AI: Prompt Engineering, LLMs, RAG, AI Agents, Semantic Search
    - Frameworks & Tools: LangChain, CrewAI, Hugging Face, OpenAI, Claude, Streamlit, REST APIs
    - Vector Databases: FAISS, ChromaDB
    - Core Programming & Web: Python, JavaScript, HTML5, CSS3, SQL
    
    WORK EXPERIENCE TIMELINE:
    1. Bluepace Tech Pvt Ltd - Generative AI Developer (March 2026 - Present)
       - Built AI-powered Dispatch Operations Assistant to streamline field service operations using LLMs & RAG.
    2. Tech Mahindra - Tech Support Engineer (June 2025 - Sept 2025)
       - Enterprise tech support, incident management, and SLA adherence.
    3. Cluck & Chuck - Founder / Marketing & Branding Specialist (Dec 2023 - June 2025)
       - Founded AI-driven cloud kitchen, using AI tools for menu creation, workflows, and branding.
    4. Trigyn Technologies - Software Engineer (AI/ML) (Nov 2019 - July 2023)
       - Built Employee Onboarding RAG Chatbot using LangChain, FAISS, and ChromaDB.
    5. Swap-Bro Technologies - Software Engineer (Nov 2019 - Dec 2020)
       - Developed interactive web UI components using JS, HTML5, CSS3.
    """

    docs = [Document(page_content=resume_content)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = setup_vector_store()
hf_client = InferenceClient(api_key=hf_token)

# 8. Session Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! I'm Shahzeb's AI Virtual Assistant. Ask me anything about his GenAI projects, overall work experience, or technical stack!", "docs": None}
    ]

# Render existing chat history
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Display saved RAG retrieved chunks if present
        if message.get("docs"):
            with st.expander("🔍 Inspect RAG Retrieval Chunks (Technical Proof)"):
                for idx, doc in enumerate(message["docs"], 1):
                    st.caption(f"**Chunk {idx}:**")
                    st.code(doc.page_content, language="text")

# 9. Quick Suggestion Pills
st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #64748b; margin-top: 1.5rem; margin-bottom: 0.5rem;'>SUGGESTED PROMPTS</p>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
quick_query = None

if c1.button("📊 Total Experience"):
    quick_query = "What is Shahzeb's overall total work experience?"
if c2.button("🚀 GenAI & RAG Projects"):
    quick_query = "What RAG and GenAI projects has Shahzeb built?"
if c3.button("🛠️ Tech Stack"):
    quick_query = "List Shahzeb's complete technical stack."
if c4.button("💼 Work Experience"):
    quick_query = "Summarize Shahzeb's career history across companies."

# 10. Handle User Input & Streaming
user_input = st.chat_input("Ask Shahzeb AI Assistant a question...") or quick_query

if user_input:
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": user_input, "docs": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate assistant response with streaming and RAG inspection
    with st.chat_message("assistant", avatar="🤖"):
        # Step A: Retrieve Context Chunks
        retrieved_docs = retriever.invoke(user_input)
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Shahzeb's AI Virtual Assistant, a professional representative for Shahzeb Adil.\n"
                    "Answer questions concisely, accurately, and enthusiastically based ONLY on the provided context.\n\n"
                    f"Context:\n{context_str}"
                )
            },
            {"role": "user", "content": user_input}
        ]

        # Step B: Stream Tokens in Real-Time
        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = hf_client.chat.completions.create(
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Step C: Render RAG Inspector Accordion
            with st.expander("🔍 Inspect RAG Retrieval Chunks (Technical Proof)"):
                for idx, doc in enumerate(retrieved_docs, 1):
                    st.caption(f"**Chunk {idx}:**")
                    st.code(doc.page_content, language="text")

            # Save to Session State
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "docs": retrieved_docs
            })

        except Exception as e:
            st.error(f"❌ Error during inference stream: {str(e)}")