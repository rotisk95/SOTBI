import subprocess
import os
import logging
import os

# Set environment variable to avoid KMeans memory leak warning on Windows
os.environ['OMP_NUM_THREADS'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
    
class LlamaInterface:
    """Interface for LLM integration using Ollama"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model_name = model_name
        self.is_initialized = False
        is_initialized = self.initialize()
        if is_initialized:
            logger.info(f"{self.model_name} initialized successfully")
        else:
            logger.error(f"Failed to initialize {self.model_name}")

    def initialize(self) -> bool:
        try:
            process = subprocess.Popen(
                ["ollama", "run", self.model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            test_prompt = "You are a teacher. Respond with 'Ready' if you understand."
            stdout, stderr = process.communicate(input=test_prompt, timeout=30)

            if process.returncode == 0 and "Ready" in stdout:
                logger.info("Ollama initialized successfully")
                return True
            else:
                logger.warning(f"Ollama initialization did not return 'Ready'. Stderr: {stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Ollama initialization timed out")
            return False
        except Exception as e:
            logger.error(f"Ollama initialization error: {str(e)}", exc_info=True)
            return False

    def get_response(self, prompt: str) -> str:
        process = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        try:
            stdout, stderr = process.communicate(input=prompt, timeout=30)
            if process.returncode == 0:
                return stdout.strip()
            else:
                logger.error(f"Ollama error: {stderr}")
                return "I apologize, but I encountered an error."
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            logger.error("Ollama response timed out")
            return "I need more time to think about this."
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}", exc_info=True)
            return "I'm having trouble processing that."
