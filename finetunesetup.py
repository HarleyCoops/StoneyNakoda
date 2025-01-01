import json
import glob
import random

def prepare_fine_tuning_data(input_dir, output_file):
    data = []
    for file in glob.glob(f"{input_dir}/*.jsonl"):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    english_word = entry['english_word']
                    
                    messages = [
                        {"role": "system", "content": "You are learning the Stoney Nakoda language. Pay close attention to language structure and offer translations or explanations as asked. You can explain the thought process in your reply."},
                        {"role": "user", "content": f"Translate and explain the Stoney version of this word: {english_word}"},
                        {"role": "assistant", "content": f"The English word '{english_word}' has the following Stoney translations:\n" + 
                         "\n".join([f"- {version['word']} ({version['grammatical_classification']}): {version['meaning']}" for version in entry['stoney_versions']])}
                    ]
                    
                    data.append({"messages": messages})
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in file {file}")
                except KeyError as e:
                    print(f"Skipping entry with missing key {e} in file {file}")

    # Add some sentence-level examples if available
    # This is just a placeholder - you'd need to create these examples
    sentence_examples = [
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant that translates English sentences to Stoney Nakoda and provides explanations."},
            {"role": "user", "content": "Translate this English sentence to Stoney: 'The sun is setting.'"},
            {"role": "assistant", "content": "The English sentence 'The sun is setting.' translates to Stoney Nakoda as:\n\nWí kȟá iyáya.\n\nExplanation:\n- Wí: sun\n- kȟá: towards\n- iyáya: to go\n\nLiterally, this translates to 'The sun is going away,' which is how the concept of sunset is expressed in Stoney Nakoda."}
        ]},
        # Add more sentence examples here in the same format
    ]
    data.extend(sentence_examples)

    random.shuffle(data)
    train_data = data[:int(len(data) * 0.8)]
    valid_data = data[int(len(data) * 0.8):]

    with open(f"{output_file}_train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(f"{output_file}_valid.jsonl", 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Training file created: {output_file}_train.jsonl")
    print(f"Validation file created: {output_file}_valid.jsonl")

prepare_fine_tuning_data("English.Data", "stoney_dictionary")