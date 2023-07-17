import os
from simpleaichat import AIChat
from rich.console import Console
import time

params = {"temperature": 0.7}  # for reproducibility
model = os.getenv("OPENAI_DEPLOYMENT_NAME")  # azure deployment name
api_url = os.getenv("OPENAI_API_BASE")  # https://xyz.openai.azure.com/
api_key = os.getenv("OPENAI_API_KEY")  # 123...
api_version = os.getenv("OPENAI_API_VERSION")  # 2023-06-01-preview
ai = AIChat(
    character="Personal assistant",
    console=False,
    params=params,
    model=model,
    api_type="azure",
    api_url=api_url,
    api_version=api_version,
)
console = Console(width=60, highlight=False)

tips = [
    "This ChatGPT model does not have access to the internet, and its training data cut-off is September 2021.",
    "ChatGPT should not be relied on for legal research of this nature, because it is very likely to invent realistic cases that do not exist.",
    "Medical and psychatric advice from ChatGPT should not be relied upon. Always consult a doctor.",
    "Tailored financial advice from ChatGPT should not be relied upon. Always consult a professional.",
    "ChatGPT is not liable for any illegal activies committed as the result of its responses.",
]

tips_prompt = """From the list of topics below, reply ONLY with the number appropriate for describing the topic of the user's message. If none are, ONLY reply with "0".

1. Content after September 2021
2. Legal/Judicial Research
3. Medical/Psychatric Advice
4. Financial Advice
5. Illegal/Unethical Activies"""

params = {
    "temperature": 0.0,
    "max_tokens": 1,
    "logit_bias": {str(k): 100 for k in range(15, 15 + len(tips) + 1)},
}

# functional
ai.new_session(
    id="tips",
    system=tips_prompt,
    save_messages=False,
    params=params,
    api_type="azure",
    api_url=api_url,
    api_version=api_version,
    api_key=api_key,
)


def check_user_input(message):
    tip_idx = ai(message, id="tips")
    if tip_idx == "0":  # no tip needed
        return
    else:
        tip = tips[int(tip_idx) - 1]
        console.print(f"⚠️ {tip}", style="bold")


while True:
    time.sleep(0.5)  # for Colab, to ensure input box appears
    try:
        user_input = console.input("[b]You:[/b] ").strip()
        if not user_input:
            break

        check_user_input(user_input)
        ai_response = ai(user_input)

        console.print(f"[b]ChatGPT[/b]: {ai_response}", style="bright_magenta")
    except KeyboardInterrupt:
        break
