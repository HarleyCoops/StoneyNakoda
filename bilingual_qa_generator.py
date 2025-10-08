# This script creates a large set of practice questions and answers for both English and Stoney languages.
# It uses Google's Gemini AI to help generate natural, meaningful questions that test
# different aspects of both languages, from basic translations to complex cultural concepts.

import json  # For working with structured data
import logging  # For keeping track of what's happening
from typing import Dict, List, Generator  # For organizing our code better
from pathlib import Path  # For handling file paths safely
from dotenv import load_dotenv  # For loading secret keys
import os  # For working with the operating system
from tqdm import tqdm  # For showing progress bars
import time  # For timing operations
from datetime import datetime  # For timestamps
import google.generativeai as genai  # Google's AI tools

# Set up our logging system to track what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BilingualQAGenerator:
    def __init__(self, english_dict_file: str, stoney_dict_file: str):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.0-pro')
        self.english_dict_file = Path(english_dict_file)
        self.stoney_dict_file = Path(stoney_dict_file)
        
        if not self.english_dict_file.exists():
            raise FileNotFoundError(f"English dictionary file not found: {english_dict_file}")
        if not self.stoney_dict_file.exists():
            raise FileNotFoundError(f"Stoney dictionary file not found: {stoney_dict_file}")

    def create_english_context_prompt(self, entries: List[Dict]) -> str:
        context = """You are an expert in the Stoney Nakoda language. Using the following English-to-Stoney dictionary entries, 
        create diverse and natural question-answer pairs that test understanding of the language.
        
        Guidelines:
        1. Create questions that test translation from English to Stoney and vice versa
        2. Focus on how multiple Stoney words can express different aspects of a single English concept
        3. Test understanding of grammatical classifications and subtle meaning differences
        4. Create scenarios that demonstrate when to use each Stoney variation
        5. Generate questions about word relationships and patterns
        6. Include cultural context where relevant
        
        Dictionary entries:
        """
        
        for entry in entries:
            context += f"\n{json.dumps(entry, ensure_ascii=False)}"
            
        return context

    def create_stoney_context_prompt(self, entries: List[Dict]) -> str:
        context = """You are an expert in the Stoney Nakoda language. Using the following Stoney-to-English dictionary entries, 
        create diverse and natural question-answer pairs that test understanding of the language.
        
        Guidelines:
        1. Create questions that test translation from Stoney to English
        2. Focus on proper usage of Stoney words in different contexts
        3. Test understanding of parts of speech and grammatical rules
        4. Create scenarios for practical usage
        5. Generate questions about cultural significance where relevant
        6. Include questions about related words and concepts
        
        Dictionary entries:
        """
        
        for entry in entries:
            context += f"\n{json.dumps(entry, ensure_ascii=False)}"
            
        return context

    def generate_qa_pairs(self, dictionary_file: Path, is_english: bool, context_size: int = 5) -> Generator[Dict, None, None]:
        entries_buffer = []
        
        with open(dictionary_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {'English' if is_english else 'Stoney'} entries"):
                try:
                    entry = json.loads(line.strip())
                    entries_buffer.append(entry)
                    
                    if len(entries_buffer) >= context_size:
                        context = self.create_english_context_prompt(entries_buffer) if is_english else self.create_stoney_context_prompt(entries_buffer)
                        prompt = """Based on these dictionary entries, generate 5 diverse 
                        question-answer pairs. Format your response EXACTLY as shown below, maintaining
                        valid JSON structure:

                        [
                            {
                                "question": "What is the Stoney word for X?",
                                "answer": "The Stoney word for X is Y."
                            }
                        ]

                        Ensure your response is a valid JSON array containing exactly 5 question-answer pairs.
                        Do not include any additional text or formatting."""
                        
                        try:
                            logger.info(f"Sending request to Google API with {len(entries_buffer)} entries...")
                            response = self.model.generate_content(
                                contents=context + "\n" + prompt
                            )
                            logger.info("Received response from Google API.")
                            response_text = response.text.strip()
                            if not response_text.startswith('['):
                                response_text = response_text[response_text.find('['):]
                            if not response_text.endswith(']'):
                                response_text = response_text[:response_text.rfind(']')+1]
                            
                            qa_pairs = json.loads(response_text)
                            for qa_pair in qa_pairs:
                                if isinstance(qa_pair, dict) and 'question' in qa_pair and 'answer' in qa_pair:
                                    qa_pair['source_language'] = 'english' if is_english else 'stoney'
                                    yield qa_pair
                                else:
                                    logger.warning("Skipping invalid QA pair format")
                        except Exception as e:
                            logger.warning(f"Error generating Q&A pairs: {str(e)}")
                            continue
                        
                        entries_buffer = entries_buffer[-2:]
                        
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line")
                    continue

    def generate_training_set(self, output_file: str, pairs_per_language: int = 75000):
        output_path = Path(output_file)
        checkpoint_dir = output_path.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        total_pairs = pairs_per_language * 2
        pair_count = 0
        checkpoint_count = 0
        start_time = time.time()
        
        logger.info(f"Starting generation of {total_pairs} Q&A pairs ({pairs_per_language} per language)...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Generate English-perspective Q&A pairs
            logger.info("Generating English-perspective Q&A pairs...")
            english_count = 0
            for qa_pair in self.generate_qa_pairs(self.english_dict_file, True):
                if english_count >= pairs_per_language:
                    break
                qa_pair['generated_at'] = datetime.now().isoformat()
                qa_pair['pair_id'] = pair_count + 1
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                pair_count += 1
                english_count += 1
                
                if pair_count % 1000 == 0:
                    self._create_checkpoint(checkpoint_dir, checkpoint_count, pair_count, total_pairs)
                    checkpoint_count += 1
            
            # Generate Stoney-perspective Q&A pairs
            logger.info("Generating Stoney-perspective Q&A pairs...")
            stoney_count = 0
            for qa_pair in self.generate_qa_pairs(self.stoney_dict_file, False):
                if stoney_count >= pairs_per_language:
                    break
                qa_pair['generated_at'] = datetime.now().isoformat()
                qa_pair['pair_id'] = pair_count + 1
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                pair_count += 1
                stoney_count += 1
                
                if pair_count % 1000 == 0:
                    self._create_checkpoint(checkpoint_dir, checkpoint_count, pair_count, total_pairs)
                    checkpoint_count += 1

        logger.info(f"Generation completed. Total time: {time.time() - start_time:.2f} seconds")

    def _create_checkpoint(self, checkpoint_dir: Path, checkpoint_count: int, pair_count: int, total_pairs: int):
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_count}.jsonl"
        with open(checkpoint_file, 'w', encoding='utf-8') as cf:
            cf.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'pairs_generated': pair_count,
                'target_pairs': total_pairs,
                'percent_complete': (pair_count / total_pairs) * 100
            }, ensure_ascii=False) + '\n')

def main():
    try:
        # Set up our file paths
        english_dict_path = "Dictionaries/english_dictionary.jsonl"
        stoney_dict_path = "Dictionaries/stoney_dictionary.jsonl"
        output_path = "Dictionaries/bilingual_training_set.jsonl"
        
        # Create our question generator
        generator = BilingualQAGenerator(english_dict_path, stoney_dict_path)
        
        # Generate all the questions and answers
        logger.info("Starting full training set generation...")
        generator.generate_training_set(output_path, pairs_per_language=75000)
        
        logger.info("Training set generation completed successfully")
                
    except Exception as e:
        logger.error(f"Error during training set generation: {str(e)}")

if __name__ == "__main__":
    main()
