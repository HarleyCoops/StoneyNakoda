import json
import os

# Get the project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_file_format(input_file, output_file):
    """Convert existing JSONL format to OpenAI fine-tuning format."""
    converted_data = []
    
    print(f"Converting {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Load the original data
                    entry = json.loads(line.strip())
                    
                    # Check if already in messages format
                    if "messages" in entry:
                        converted_data.append(entry)
                    else:
                        # Convert from old format
                        converted_entry = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You have been finetuned on the entire Stoney dictionary. Concisely Translate or explain Stoney words or concepts or how English words are said in Stoney."
                                },
                                {
                                    "role": "user",
                                    "content": entry["text"]
                                },
                                {
                                    "role": "assistant",
                                    "content": entry["labels"]["output"]
                                }
                            ]
                        }
                        converted_data.append(converted_entry)
                    
                except json.JSONDecodeError:
                    print(f"Error parsing line {line_num}")
                except KeyError as e:
                    print(f"Missing key in line {line_num}: {e}")
    
    except FileNotFoundError:
        print(f"Could not find file: {input_file}")
        return
    
    # Write the converted data with special character handling
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            # Use ensure_ascii=False to preserve special characters
            # Use separators to minimize whitespace
            json_str = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
            f.write(json_str + '\n')
    
    print(f"Converted {len(converted_data)} entries to {output_file}")

def print_sample_data(file_path, num_samples=3):
    """Print sample entries from a JSONL file."""
    print(f"\nSample data from {file_path}:")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()[:num_samples]
            for i, line in enumerate(lines, 1):
                # Parse and pretty print while preserving special characters
                entry = json.loads(line)
                print(f"\nSample {i}:")
                print(json.dumps(entry, ensure_ascii=False, indent=2))
    except FileNotFoundError:
        print(f"Could not find file: {file_path}")

if __name__ == "__main__":
    # Define paths relative to project root
    train_input = os.path.join(PROJECT_ROOT, "data", "raw", "NOMIC_stoney_train.jsonl")
    val_input = os.path.join(PROJECT_ROOT, "data", "raw", "NOMIC_stoney_val.jsonl")
    
    train_output = os.path.join(PROJECT_ROOT, "data", "processed", "stoney_train_formatted.jsonl")
    val_output = os.path.join(PROJECT_ROOT, "data", "processed", "stoney_val_formatted.jsonl")
    
    # Convert both files
    convert_file_format(train_input, train_output)
    convert_file_format(val_input, val_output)
    
    # Print samples from the converted files
    print_sample_data(train_output)
    print_sample_data(val_output)
    
    print("\nNext steps:")
    print("1. Review the sample data above to ensure it looks correct")
    print("2. If the format looks good, use these new files with openai_finetune.py")
    print("3. Update the file paths in openai_finetune.py to use the new *_formatted.jsonl files") 