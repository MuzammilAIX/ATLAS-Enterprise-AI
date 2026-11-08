
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

# Load environment variables
load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

if not API_KEY:
    raise ValueError(f"OPENAI_API_KEY missing in {ENV_PATH}")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

try:
    print("Checking OpenAI API connection...")

    response = client.responses.create(
        model=MODEL,
        input="Hello! Reply with ATLAS connected."
    )

    print("\nOpenAI API connected successfully!")
    print("Model:", MODEL)
    print("Response:", response.output_text)

except APIStatusError as error:
    print("\nOpenAI API Error")
    print("Status:", error.status_code)
    print("Details:", error.response.text)

except Exception as error:
    print("\nError:", str(error))
