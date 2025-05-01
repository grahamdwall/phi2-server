# mortgage_llm.py

from openai import OpenAI
import gradio as gr
import os
import re

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

session_data = {
    "house_price": None,       # full property cost
    "down_payment": 0,         # default 0 if not mentioned
    "principal": None,         # house_price - down_payment
    "rate": None,              # interest rate
    "years": None,             # loan term
}

# --- Real LLM function to call OpenAI Chat Completion ---
def ask_llm(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content

def calculate_weekly_mortgage(principal, annual_rate, years):
    weekly_rate = (1 + (annual_rate / 100)) ** (1/52) - 1
    num_weeks = years * 52
    if weekly_rate == 0:
        weekly_payment = principal / num_weeks
    else:
        weekly_payment = principal * (weekly_rate * (1 + weekly_rate) ** num_weeks) / ((1 + weekly_rate) ** num_weeks - 1)
    return round(weekly_payment, 2)

# Check if conversation mentions all three and trigger calculation
def should_calculate(history):
    text = " ".join([m["content"].lower() for m in history if m["role"] == "user"])
    return all(keyword in text for keyword in ["$", "%", "year", "loan", "term"])

def clarify_input(user_input):
    """Detect ambiguous single numbers and request clarification."""
    numbers = re.findall(r'\d+(\.\d+)?', user_input)
    if numbers:
        if len(numbers) == 1:
            num = float(numbers[0])
            if 0 < num < 20:
                return "You entered **{}**. Is that your interest rate (%)?".format(num)
            elif 20 <= num <= 50:
                return "You entered **{}**. Is that the loan term (years)?".format(int(num))
            else:
                return "You entered **${:,}**. Is that the full house price or your down payment?".format(int(num))
    return None

def extract_principal(history):
    """Extract the loan principal ($ amount) from the conversation history."""
    for message in history[::-1]:  # Search from newest to oldest
        text = message["content"]
        matches = re.findall(r'\$\s?([\d,]+)', text)
        if matches:
            amount = matches[0].replace(',', '')
            return float(amount)
    return None

def extract_rate(history):
    """Extract the interest rate (%) from the conversation history."""
    for message in history[::-1]:
        text = message["content"]
        matches = re.findall(r'(\d+(\.\d+)?)\s?%', text)
        if matches:
            return float(matches[0][0])
    return None

def extract_years(history):
    """Extract the loan term (years) from the conversation history."""
    for message in history[::-1]:
        text = message["content"].lower()
        match = re.search(r'(\d+)\s*(years|year)', text)
        if match:
            return int(match.group(1))
    return None

def extract_down_payment(history):
    """Extract down payment if mentioned ($ amount)."""
    for message in history[::-1]:
        text = message["content"]
        matches = re.findall(r'down payment.*?\$?([\d,]+)', text, re.IGNORECASE)
        if matches:
            amount = matches[0].replace(',', '')
            return float(amount)
    return None

def format_history(history):
    formatted = []
    for message in history:
        # Gradio passes history as [{"role": ..., "content": ...}] already
        if isinstance(message, dict):
            formatted.append(message)
        elif isinstance(message, list) and len(message) == 2:
            # If still receiving old tuple format, convert it
            formatted.append({"role": "user", "content": message[0]})
            formatted.append({"role": "assistant", "content": message[1]})
    return formatted

def mortgage_assistant(user_message, history):
    # Prepare the conversation for the LLM
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful mortgage payment calculator located in Ontario, Canada acting as an assistant to real estate agent Sue. "
            "Return ONLY a single factual sentence when enough data is available. "
            "Use this format: "
            "'The monthly mortgage payment for a $XXX,XXX loan with an interest rate of X.XX% and a loan term of XX years is approximately $X,XXX.XX.' "
            "Do not greet, explain, or offer assistance when you have enough data. "
            "But if you don't have enough information, ask the user to provide it clearly."
            "If the user asks for information around mortgages, but not specifically about payments, then refer them to Sue"
        )
    }
    formatted_history = format_history(history)
    conversation = [system_message] + formatted_history + [{"role": "user", "content": user_message}]

    # Try to extract mortgage information
    extracted_principal = extract_principal(formatted_history + [{"role": "user", "content": user_message}])
    extracted_rate = extract_rate(formatted_history + [{"role": "user", "content": user_message}])
    extracted_years = extract_years(formatted_history + [{"role": "user", "content": user_message}])
    extracted_down_payment = extract_down_payment(formatted_history + [{"role": "user", "content": user_message}])

    # If ALL 3 are found, trigger calculation immediately!
    if extracted_principal is not None and extracted_rate is not None and extracted_years is not None:
        if extracted_down_payment is not None:
            principal = extracted_principal - extracted_down_payment
        else:
            principal = extracted_principal
        weekly_payment = calculate_weekly_mortgage(principal, extracted_rate, extracted_years)
        
        # Inject the real answer directly!
        assistant_message = (
            f"Your weekly payment for a **${extracted_principal:,.0f}** loan at **{extracted_rate}%** over **{extracted_years} years** "
            f"is estimated to be **${weekly_payment}**."
        )
    else:
            # Still missing info → politely ask
            assistant_message = ask_llm(conversation)
    
    return assistant_message

# --- Launch Gradio Chat Interface ---
demo = gr.ChatInterface(
    fn=mortgage_assistant,
    title="🏡 Smart Mortgage Assistant (LLM-powered)",
    description="Talk to an AI that helps you calculate mortgage payments intelligently!",
    theme="default",
    examples=["I need a mortgage for $350,000", "5% interest over 30 years"],
)

demo.launch()
