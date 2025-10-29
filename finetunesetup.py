import json
import random
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_fine_tuning_data(input_file: str, output_dir: str):
    """
    Converts a JSONL file of Q&A pairs to the OpenAI fine-tuning format,
    then splits it into training and validation sets.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_train_file = os.path.join(output_dir, "stoney_train.jsonl")
    output_valid_file = os.path.join(output_dir, "stoney_valid.jsonl")

    data = []
    logger.info(f"Reading and converting data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    question = entry.get("question")
                    answer = entry.get("answer")
                    
                    if not question or not answer:
                        logger.warning(f"Skipping entry with missing 'question' or 'answer': {entry}")
                        continue

                    messages = [
                        {"role": "system", "content": "You are a bilingual Stoney-English assistant. You have been fine-tuned on a comprehensive set of Stoney Nakoda language data. Your purpose is to provide accurate translations, explain grammatical concepts, and offer cultural context when appropriate. Respond concisely and accurately."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                    data.append({"messages": messages})
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {input_file}")
                except KeyError as e:
                    logger.warning(f"Skipping entry with missing key {e} in {input_file}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return

    if not data:
        logger.error("No data was processed. Exiting.")
        return

    # Shuffle and split the data
    logger.info(f"Successfully converted {len(data)} entries. Shuffling and splitting data...")
    random.shuffle(data)
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    valid_data = data[split_index:]

    # Write training data
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(train_data)} lines to training file: {output_train_file}")

    # Write validation data
    with open(output_valid_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(valid_data)} lines to validation file: {output_valid_file}")

    logger.info("Data preparation complete.")

if __name__ == "__main__":
    input_qa_file = "Dictionaries/bilingual_training_set_v2.jsonl"
    output_directory = "OpenAIFineTune/"
    prepare_fine_tuning_data(input_qa_file, output_directory)
