import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re

# -----------------------------
# 1️⃣ Utility functions
# -----------------------------
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
        return f"""ROLE:
You are an expert in cold email copywriting, specializing in persuasive follow-up communication.

TASK:
Write a follow-up cold outreach email (<150 words) for the prospect.
The goal is to re-engage them after the previous email and secure a meeting
to discuss comprehensive {email_type} compliance and solutions offered by our NGO.

YOUR ORGANIZATION:
- NGO specializing in {email_type} compliance services
- Offering end-to-end {email_type} solutions

TARGET PROSPECT DETAILS:
- Name: {prospect.get('prospect_name', 'N/A')}
- Title: {prospect.get('title', 'N/A')}
- Company: {prospect.get('company', 'N/A')}
- Industry: {prospect.get('industry', 'N/A')}
- Location: {prospect.get('location', 'N/A')}
- LinkedIn: {prospect.get('prospect_linkedin', 'N/A')}
- Experience: {prospect.get('experience', 'N/A')}
- Specialties: {prospect.get('specialties', 'N/A')}
- Network Size: {prospect.get('network_size', 'N/A')}
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
- Acknowledge the previous email politely (“I shared a note earlier…”).
- Keep it concise, polite, and respectful of their time.
- Provide a light nudge or reminder of the value you bring.
- End with a soft call-to-action (“Would you be open for a quick chat next week?”).
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
- Location: {prospect.get('location', 'N/A')}
- LinkedIn: {prospect.get('prospect_linkedin', 'N/A')}
- Experience: {prospect.get('experience', 'N/A')}
- Specialties: {prospect.get('specialties', 'N/A')}
- Network Size: {prospect.get('network_size', 'N/A')}
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
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

        # Skip the input part
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return clean_json_response(response, prompt)

    except Exception as e:
        return {
            "subject": "Error generating email",
            "body": f"Error occurred: {str(e)}",
            "signature": "[Your Name]\n[Your Organization]"
        }

# -----------------------------
# 2️⃣ Load model & adapter
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(base_model_path, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",  # Handles GPU allocation automatically
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model.to(device)
    return tokenizer, model, device

# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Email Generator", layout="wide")
st.title("🌟 Personalized Email Generator")

base_model_path = "./base_model"       
adapter_path = "./fine_tuned_adapter" 

tokenizer, model, device = load_model(base_model_path, adapter_path)

st.sidebar.header("Prospect Details")
prospect_name = st.sidebar.text_input("Prospect Name")
title = st.sidebar.text_input("Title")
company = st.sidebar.text_input("Company")
industry = st.sidebar.text_input("Industry")
linkedin = st.sidebar.text_input("LinkedIn")
specialties = st.sidebar.text_input("Specialties")
linkedin_summary = st.sidebar.text_area("LinkedIn Summary", height=150)
product_service = st.sidebar.text_area("Product / Service", height=150)
key_differentiators = st.sidebar.text_area("Key Differentiators", height=150)
case_studies = st.sidebar.text_area("Case Studies", height=150)
industry_insights = st.sidebar.text_area("Industry Insights", height=150)
pain_points = st.sidebar.text_area("Pain Points", height=150)
industry_usecase = st.sidebar.text_area("Industry Use Case", height=150)
previous_email = st.sidebar.text_area("Previous Email (for follow-up)", height=150)

prospect_data = {
    "prospect_name": prospect_name,
    "title": title,
    "company": company,
    "industry": industry,
    "prospect_linkedin": linkedin,
    "specialties": specialties,
    "linkedin_summary": linkedin_summary,
    "product_service": product_service,
    "key_differentiators": key_differentiators,
    "case_studies": case_studies,
    "industry_insights": industry_insights,
    "pain_points": pain_points,
    "industry_usecase": industry_usecase
}

email_type = st.sidebar.selectbox("Email Type", ["POSH", "CSR"])

if st.button("Generate Email"):
    with st.spinner("Generating personalized email..."):
        email_output = generate_email(prospect_data, tokenizer, model, email_type=email_type, previous_email=previous_email, device=device)

        st.subheader("📬 Generated Email")
        st.markdown(f"**Subject:** {email_output['subject']}")
        st.markdown(f"**Body:**\n{email_output['body']}")
        st.markdown(f"**Signature:**\n{email_output.get('signature', '[Your Name]\\n[Your Organization]')}")
