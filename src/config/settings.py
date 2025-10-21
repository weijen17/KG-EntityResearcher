"""Configuration settings for the two-agent system"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model Configuration
    MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "openai")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    # Agent Configuration
    RESEARCH_AGENT_VERBOSE: bool = True
    ANALYST_AGENT_VERBOSE: bool = True

    def __init__(self):
        """Initialize settings and create necessary directories"""
        self.OUTPUT_DIR.mkdir(exist_ok=True)

        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")


settings = Settings()

