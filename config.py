import os


# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# Application configuration
APP_NAME = "University Admission AI"

MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.2


def validate_configuration():
    """Check that required environment variables are available."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    return True
