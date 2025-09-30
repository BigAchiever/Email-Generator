
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re
from datetime import datetime

# -----------------------------Utility functions-----------------------------
def clean_json_response(response, original_prompt=""):
    try:
        if original_prompt and original_prompt.strip() in response:
            response = response.replace(original_prompt.strip(), "").strip()

        json_patterns = [
            r'\{\s*"subject"\s*:\s*"[^"]*"\s*,\s*"body"\s*:\s*"[^"]*"\s*,\s*"signature"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"subject"\s*:.*?"signature"\s*:.*?\}',
            r'\{[^{}]*"subject"[^{}]*"body"[^{}]*"signature"[^{}]*\}',
            r'\{.*?"subject".*?"body".*?"signature".*?\}'
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                json_str = matches[0]
                json_str = re.sub(r'[\n\r\t]+', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    parsed = json.loads(json_str)
                    if all(key in parsed for key in ['subject', 'body', 'signature']):
                        return parsed
                except json.JSONDecodeError:
                    continue

        return extract_fallback(response)

    except Exception:
        return extract_fallback(response)


def extract_fallback(response):
    subject_patterns = [r'"subject"\s*:\s*"([^"]*)"', r'Subject:\s*([^\n]*)']
    body_patterns = [r'"body"\s*:\s*"([^"]*)"', r'Body:\s*(.*?)(?=Signature:|"signature"|$)']
    signature_patterns = [r'"signature"\s*:\s*"([^"]*)"', r'Signature:\s*(.*?)$']

    def extract_with_patterns(patterns, text):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().strip('"').strip("'")
        return None

    subject = extract_with_patterns(subject_patterns, response) or "Partnership Discussion"
    body = extract_with_patterns(body_patterns, response) or "I'd like to discuss potential collaboration opportunities."
    signature = extract_with_patterns(signature_patterns, response) or "[Your Name]\n[Your Organization]"

    return {"subject": subject, "body": body, "signature": signature}


def build_prompt(prospect, email_type="CSR", previous_email=None):
    if previous_email:
        return f"""
        ROLE: You are an expert cold email copywriter specializing in polite, persuasive **follow-up communication**.

        TASK:
        Write a **follow-up cold email (<150 words)** for the prospect.
        Your job is to re-engage them by referencing the **previous email**, building on its key points, and nudging them toward a short conversation.

        YOUR ORGANIZATION:
        - NGO specializing in {email_type} compliance services
        - Providing end-to-end {email_type} solutions (advisory, training, audits, implementation)

        TARGET PROSPECT DETAILS:
        - Name: {prospect.get('prospect_name', 'N/A')}
        - Title: {prospect.get('title', 'N/A')}
        - Company: {prospect.get('company', 'N/A')}
        - Industry: {prospect.get('industry', 'N/A')}
        - LinkedIn: {prospect.get('prospect_linkedin', 'N/A')}
        - LinkedIn Personal Summary: {prospect.get('linkedin_summary', 'N/A')}

        COMPANY CONTEXT ({prospect.get('company', '')}):
        - Company Context: {prospect.get('company_context', 'N/A')}
        - Product/Service Description: {prospect.get('product_service', 'N/A')}
        - Key Differentiators: {prospect.get('key_differentiators', 'N/A')}
        - Case Studies: {prospect.get('case_studies', 'N/A')}
        - Industry Insights: {prospect.get('industry_insights', 'N/A')}
        - Pain Points: {prospect.get('pain_points', 'N/A')}
        - Industry-Specific Use Case: {prospect.get('industry_usecase', 'N/A')}

        PREVIOUS EMAIL SENT:
        \"\"\"{previous_email}\"\"\"

        FOLLOW-UP RULES:
        - Politely acknowledge the earlier note (“I shared a message last week…”).
        - Highlight ONE specific value or benefit from the previous email (don’t repeat everything).
        - Add ONE fresh angle (industry trend, compliance risk, or company value alignment).
        - Keep it brief, respectful, and professional.
        - Close with a light, soft call-to-action (e.g., “Would you be open for a quick 15-min chat?”).
        - Tone: warm, professional, persistent but not pushy.

        DELIVERABLE:
        Return ONLY valid JSON with: {{"subject": "...", "body": "...", "signature": "..."}}"""
    else:
        return f"""ROLE:
        You are an expert in copywriting and marketing, specializing in persuasive, high-response email communication.

        TASK:
        Craft 1 personalized cold outreach email (<200 words) for the prospect, with the goal of securing a meeting to discuss comprehensive {email_type} compliance and solutions offered by our NGO.

        YOUR ORGANIZATION:
        - NGO specializing in {email_type} compliance services
        - Offering end-to-end {email_type} solutions

        TARGET PROSPECT DETAILS:
        - Name: {prospect.get('prospect_name', 'N/A')}
        - Title: {prospect.get('title', 'N/A')}
        - Company: {prospect.get('company', 'N/A')}
        - Industry: {prospect.get('industry', 'N/A')}
        - LinkedIn: {prospect.get('prospect_linkedin', 'N/A')}
        - LinkedIn Personal Summary: {prospect.get('linkedin_summary', 'N/A')}

        COMPANY CONTEXT ({prospect.get('company', '')}):
        - Company Context: {prospect.get('company_context', 'N/A')}
        - Product/Service Description: {prospect.get('product_service', 'N/A')}
        - Key Differentiators: {prospect.get('key_differentiators', 'N/A')}
        - Case Studies: {prospect.get('case_studies', 'N/A')}
        - Industry Insights: {prospect.get('industry_insights', 'N/A')}
        - Pain Points: {prospect.get('pain_points', 'N/A')}
        - Industry-Specific Use Case: {prospect.get('industry_usecase', 'N/A')}

        PERSONALIZATION RULES:
        - Reference company focus, initiatives, recognition, or products/services.
        - Reference prospect's role, expertise, LinkedIn summary, or contributions.
        - Emails must feel warm, professional, and research-driven — avoid generic language.

        EMAIL STRUCTURE:
        - Subject line: short, personalized, curiosity-driven.
        - First 3 lines: strong hook (based on company recognition, initiatives, products, or prospect role).
        - Middle: connect NGO's {email_type} expertise with their culture/values.
        - Closing: soft, clear call-to-action.
        - Keep tone friendly, forward-looking, professional.

        DELIVERABLE:
        Return ONLY valid JSON with: {{"subject": "...", "body": "...", "signature": "..."}}"""


def generate_email(prospect_data, tokenizer, model, email_type="POSH", previous_email=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    prompt = build_prompt(prospect_data, email_type, previous_email=previous_email)
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt

        inputs = tokenizer(
            formatted_prompt, return_tensors="pt",
            truncation=True, max_length=2048, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                min_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return clean_json_response(response, prompt)

    except Exception as e:
        return {
            "subject": "Error generating email",
            "body": f"Error occurred: {str(e)}",
            "signature": "[Your Name]\n[Your Organization]"
        }

# ---------------------------------------------------Loading model & adapter -----------------------------------------------------------
# 
@st.cache_resource(show_spinner=True)
def load_model(base_model_path, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return tokenizer, model, device

#  CSS 
# -----------------------------
def load_css():
    st.markdown("""
    <style>
    /* Import a classic font */
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600&family=Nunito+Sans:wght@400;600&display=swap');

    /* Main layout */
    .main {
        background-color: #f8f9fa; /* Off-white background */
        font-family: 'Nunito Sans', sans-serif;
        color: #343a40; /* Dark charcoal text for readability */
    }

    /* Headings - Using a classic serif font */
    h1, h2, h3 {
        font-family: 'Lora', serif;
        font-weight: 600;
        color: #2c3e50; /* Deep slate blue */
    }
    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.5rem; margin-top: 1.8rem; }

    /* Email preview container */
    .email-preview {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04); /* Softer shadow */
    }

    /* Subject line */
    .email-subject {
        font-family: 'Lora', serif;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 1rem;
        color: #343a40;
    }

    /* Body text */
    .email-body {
        color: #495057; /* Slightly softer than pure black */
        line-height: 1.7;
        white-space: pre-wrap;
        margin: 1rem 0;
        font-size: 1rem;
    }

    /* Signature */
    .email-signature {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d; /* Muted gray */
        font-size: 0.9rem;
    }

    /* Messages */
    .info-msg, .warning-msg {
        padding: 12px 18px;
        border-radius: 6px;
        border-width: 1px;
        border-style: solid;
        margin-bottom: 15px;
        font-weight: 400;
    }
    .info-msg {
        background-color: #eef2f7;
        color: #4a6a8a;
        border-color: #d2dce6;
    }
    .warning-msg {
        background-color: #fef8e7;
        color: #8a6c3d;
        border-color: #fbeac4;
    }

    /* Buttons */
    .stButton > button {
        border: 1px solid #adb5bd; /* Subtle border */
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease-in-out;
        background-color: #ffffff;
        color: #495057;
    }
    .stButton > button:hover {
        background-color: #e9ecef !important; /* Gentle hover effect */
        color: #212529 !important;
        border-color: #adb5bd !important;
    }

    /* Primary button style */
    .stButton:has(button[kind="primary"]) > button {
        background-color: #2c3e50; /* Deep slate blue */
        color: white;
        border-color: #2c3e50;
    }
    .stButton:has(button[kind="primary"]) > button:hover {
        background-color: #46617c !important;
        border-color: #46617c !important;
        color: white !important;
    }

    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="Email Generator",
    page_icon="📧",
    layout="wide"
)

load_css()

# session state
if 'email_history' not in st.session_state:
    st.session_state.email_history = []
if 'current_email' not in st.session_state:
    st.session_state.current_email = None

# Header 
st.title("📧 AI Email Generator")
st.caption("Create personalized outreach emails for compliance campaigns")

#  -----------------------------------------------------------------Model loading--------------------------------------------------------------------
base_model_path = "PATH TO YOUR BASE MODEL"
adapter_path = "PATH TO YOUR ADAPTER"

with st.spinner("Loading AI model..."):
    tokenizer, model, device = load_model(base_model_path, adapter_path)

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    email_type = st.selectbox("Campaign Type", ["POSH", "CSR"])

    st.divider()

    st.metric("Emails Generated", len(st.session_state.email_history))

    if st.button("Clear History", type="secondary", use_container_width=True):
        st.session_state.email_history = []
        st.rerun()

#  ----------------------------------------------------------Streamlit tabs-----------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 Create", "📜 History"])

with tab1:
    st.subheader("Prospect Details")

    col1, col2 = st.columns(2)
    with col1:
        prospect_name = st.text_input("Name", placeholder="John Smith")
        title = st.text_input("Job Title", placeholder="VP of Operations")
    with col2:
        company = st.text_input("Company", placeholder="TechCorp Inc.")
        industry = st.text_input("Industry", placeholder="Technology")

    linkedin = st.text_input("LinkedIn URL (optional)", placeholder="https://linkedin.com/in/...")
    linkedin_summary = st.text_area("LinkedIn Summary", placeholder="Professional bio or about section...", height=120)

    st.divider()

    st.subheader("Company Information")
    col3, col4 = st.columns(2)
    with col3:
        product_service = st.text_area("Products / Services", height=120)
        key_differentiators = st.text_area("Key Differentiators", height=120)
    with col4:
        pain_points = st.text_area("Pain Points", height=120)
        case_studies = st.text_area("Case Studies", height=120)

    industry_insights = st.text_area("Industry Insights", height=100)
    industry_usecase = st.text_area("Industry Use Case", height=100)

    st.divider()

    is_followup = st.checkbox("This is a follow-up email")
    previous_email = None
    if is_followup:
        st.markdown('<div class="info-msg">Include the previous email for context</div>', unsafe_allow_html=True)
        previous_email = st.text_area("Previous Email", height=120)

    # Validation
    required_fields = [prospect_name, title, company, industry]
    all_filled = all(field.strip() for field in required_fields)

    if not all_filled:
        st.markdown('<div class="warning-msg">⚠️ Please fill all required fields</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("Generate Email", type="primary", disabled=not all_filled, use_container_width=True):
        prospect_data = {
            "prospect_name": prospect_name,
            "title": title,
            "company": company,
            "industry": industry,
            "prospect_linkedin": linkedin,
            "linkedin_summary": linkedin_summary,
            "product_service": product_service,
            "key_differentiators": key_differentiators,
            "case_studies": case_studies,
            "industry_insights": industry_insights,
            "pain_points": pain_points,
            "industry_usecase": industry_usecase
        }

        with st.spinner("Generating email... (~15 seconds)"):
            email_output = generate_email(
                prospect_data,
                tokenizer,
                model,
                email_type=email_type,
                previous_email=previous_email if is_followup else None,
                device=device
            )

            st.session_state.current_email = email_output
            st.session_state.email_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prospect": prospect_name,
                "company": company,
                "email_type": email_type,
                "is_followup": is_followup,
                "email": email_output
            })

        st.success("✅ Email generated!")
        st.rerun()

    # Display generated email
    if st.session_state.current_email:
        st.divider()
        st.subheader("Generated Email")

        email = st.session_state.current_email

        st.markdown(f"""
        <div class="email-preview">
            <div class="email-subject">Subject: {email['subject']}</div>
            <div class="email-body">{email['body']}</div>
            <div class="email-signature">{email.get('signature', '[Your Name]<br>[Your Organization]')}</div>
        </div>
        """, unsafe_allow_html=True)

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            email_text = f"Subject: {email['subject']}\n\n{email['body']}\n\n{email.get('signature', '[Your Name], [Your Organization]')}"
            st.download_button(
                "💾 Download",
                data=email_text,
                file_name=f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
        with col_a2:
            if st.button("📋 Copy Text", use_container_width=True):
                st.code(email_text, language=None)
        with col_a3:
            if st.button("🔄 New Version", use_container_width=True):
                st.session_state.current_email = None
                st.rerun()

with tab2:
    st.subheader("Email History")

    if st.session_state.email_history:
        for idx, entry in enumerate(reversed(st.session_state.email_history)):
            with st.expander(
                f"{entry['prospect']} @ {entry['company']} - {entry['timestamp']}",
                expanded=(idx==0)
            ):
                st.caption(f"{entry['email_type']} • {'Follow-up' if entry['is_followup'] else 'Initial'}")
                st.markdown(f"**Subject:** {entry['email']['subject']}")
                st.text_area("Body", entry['email']['body'], height=150, key=f"hist_body_{idx}", disabled=True)
                st.text_area("Signature", entry['email'].get('signature', ''), height=60, key=f"hist_sig_{idx}", disabled=True)
    else:
        st.info("No emails generated yet")