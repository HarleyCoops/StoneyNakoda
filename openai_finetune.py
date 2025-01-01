import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OpenAIFineTuner:
    def __init__(self):
        """Initialize the OpenAI fine-tuner with API key and file paths."""
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        
        # Define file paths using absolute paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_file = os.path.join(self.base_dir, "OpenAIFineTune", "stoney_train.jsonl")
        self.valid_file = os.path.join(self.base_dir, "OpenAIFineTune", "stoney_valid.jsonl")
        
        # Ensure files exist
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        if not os.path.exists(self.valid_file):
            raise FileNotFoundError(f"Validation file not found: {self.valid_file}")
        
        logger.info(f"Found training file: {self.train_file}")
        logger.info(f"Found validation file: {self.valid_file}")

    def upload_file(self, file_path: str, purpose: str) -> str:
        """Upload a file to OpenAI and return its file ID."""
        logger.info(f"Uploading {purpose} file: {file_path}")
        
        with open(file_path, 'rb') as file:
            response = self.client.files.create(
                file=file,
                purpose=purpose
            )
        
        logger.info(f"Successfully uploaded {purpose} file. File ID: {response.id}")
        return response.id

    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: str) -> str:
        """Create a fine-tuning job and return its ID."""
        logger.info("Creating fine-tuning job...")
        
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model="gpt-3.5-turbo",  # You can change this to gpt-4 if needed
            hyperparameters={
                "n_epochs": 3  # Adjust as needed
            }
        )
        
        logger.info(f"Fine-tuning job created successfully. Job ID: {response.id}")
        return response.id

    def monitor_job_progress(self, job_id: str, check_interval: int = 60):
        """Monitor the progress of a fine-tuning job."""
        logger.info(f"Starting to monitor fine-tuning job: {job_id}")
        
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            # Log detailed status information
            logger.info(f"Status: {status}")
            if hasattr(job, 'trained_tokens'):
                logger.info(f"Trained tokens: {job.trained_tokens}")
            if hasattr(job, 'training_accuracy'):
                logger.info(f"Training accuracy: {job.training_accuracy}")
            if hasattr(job, 'validation_loss'):
                logger.info(f"Validation loss: {job.validation_loss}")
            
            if status == "succeeded":
                logger.info(" Fine-tuning completed successfully!")
                logger.info(f"Fine-tuned model ID: {job.fine_tuned_model}")
                break
            elif status == "failed":
                logger.error(f" Fine-tuning failed: {job.error}")
                break
            elif status in ["cancelled", "expired"]:
                logger.warning(f" Fine-tuning job {status}")
                break
            
            logger.info(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)

    def run_fine_tuning(self):
        """Run the complete fine-tuning process."""
        try:
            # Step 1: Upload files
            logger.info("Step 1/3: Uploading files to OpenAI")
            train_file_id = self.upload_file(self.train_file, "fine-tune")
            valid_file_id = self.upload_file(self.valid_file, "fine-tune")
            
            # Step 2: Create fine-tuning job
            logger.info("Step 2/3: Creating fine-tuning job")
            job_id = self.create_fine_tuning_job(train_file_id, valid_file_id)
            
            # Step 3: Monitor progress
            logger.info("Step 3/3: Monitoring fine-tuning progress")
            self.monitor_job_progress(job_id)
            
        except Exception as e:
            logger.error(f" Error during fine-tuning process: {str(e)}")
            raise

def main():
    """Main function to run the fine-tuning process."""
    logger.info("=== Starting Stoney Language Model Fine-Tuning ===")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        tuner = OpenAIFineTuner()
        tuner.run_fine_tuning()
        
        logger.info("=== Fine-Tuning Process Completed ===")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
