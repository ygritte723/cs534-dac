import json
import os
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from torch.cuda.amp import autocast
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
import re

torch.cuda.empty_cache()
os.umask(0)

def paraphrase_sentence(input_sentence, paraphraser, num_return_sequences=3):
    """
    Generate paraphrased sentences for the given input.
    """
    outputs = paraphraser(input_sentence, max_length=50, num_return_sequences=num_return_sequences, do_sample=True)
    return [output['generated_text'] for output in outputs]

def clean_output(text):
    """
    Clean generated text to remove artifacts like tabs, enumerations, and markdown symbols.
    """
    cleaned_sentences = []
    for line in text.split("\n"):
        # Strip leading/trailing whitespace
        line = line.strip()
        # Remove enumerations (e.g., "1)", "2)", "3.") at the start of lines
        line = re.sub(r"^\d+[\)\.\t\s]+", "", line)
        # Remove markdown-like symbols (e.g., "**", "*", "#", "<DOCID>")
        line = re.sub(r"[*#<>]+", "", line)
        # Replace tabs and excessive spaces
        line = re.sub(r"\t+", " ", line)
        line = re.sub(r"\s{2,}", " ", line)
        # Skip empty or invalid lines
        if line and not line.lower().startswith(("introduction", "list of figures")):
            cleaned_sentences.append(line)
    return "\n".join(cleaned_sentences)
def filter_hallucinations(input_text, expanded_text, similarity_model, min_similarity=0.5):
    """
    Filter hallucinated sentences by checking semantic similarity with the input context.
    """
    input_embedding = similarity_model.encode(input_text, convert_to_tensor=True)
    sentences = expanded_text.split("\n")
    filtered_sentences = []

    # Batch encode sentences for similarity comparison
    sentence_embeddings = similarity_model.encode(sentences, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, sentence_embeddings).squeeze(0)

    # Get the indices of the top N most similar sentences
    top_indices = torch.topk(similarities, k=min(4, len(sentences))).indices.tolist()

    # Retrieve the top N sentences
    filtered_sentences = [sentences[idx].lstrip("1234567890\t. ") for idx in top_indices]

    return filtered_sentences

def process_file(json_file_name, model_name, tokenizer_name, similarity_model_name, device_id):
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # Load the model and tokenizer correctly in each worker
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        similarity_model = SentenceTransformer(similarity_model_name).to(device)

        tokenizer.pad_token = tokenizer.eos_token  # Set the pad_token to eos_token to avoid issues
# for json_file_name in tqdm(json_files, desc="Processing JSON files"):
        clean_json_file_name = os.path.basename(json_file_name)[:-5]
        file_path = f"../CC_10000/LLM_ha_4/{clean_json_file_name}.json"
        # print(file_path)
        

        # Read the JSON file
        with open(json_file_name) as f:
            data = json.load(f)
        short_description = data["positive_caption"][0]

        # Generate the input text
        input_text = f"short: please describe what you might see in a picture of a scene that contains 'a Christmas tree', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to\n \
                long: 	1	In the center of the room, a majestic evergreen Christmas tree stands tall, adorned with twinkling lights and colorful ornaments.\n \
        	2	Delicate strands of tinsel gracefully drape the tree's branches, adding a touch of shimmer to the festive display.\n \
        	3	An elegant star or angel graces the top of the tree, representing the Star of Bethlehem or the heavenly messengers present at Jesus' birth.\n \
        	4	Wrapped presents in various shapes and sizes are piled beneath the tree, their festive gift wrap and bows hinting at the surprises inside.\n \
        	5	A cozy fireplace crackles nearby, with stockings hung from the mantel, eagerly awaiting the arrival of Santa Claus.\n \
        	6	Lush green garlands and flickering candles decorate the mantel, enhancing the holiday atmosphere.\n \
        	7	Comfortable seating arrangements, such as sofas and armchairs, are positioned near the tree, complete with plush cushions and warm throw blankets.\n \
        	8	Family members and friends gather around the tree in festive attire, sharing laughter and conversation.\n \
        	9	A beautifully crafted wreath hangs on a nearby wall or window, adding an additional touch of holiday cheer.\n \
        	10	Through the window, a snowy winter landscape can be seen, with snow-covered trees, rooftops, and gently falling snowflakes, creating the perfect backdrop for the Christmas scene.\n \
                short: please describe what you might see in a picture of a scene that contains 'a male hand playing nervously with a pencil on a black background', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to \n \
                long:   1	A male hand is positioned prominently in the frame, with fingers flexing and shifting as they manipulate a pencil.\n \
        	2	The pencil, held between the thumb and index finger, twirls and spins as the hand moves it nervously.\n \
        	3	Shadows from the hand and pencil cast dramatic patterns on the stark black background, emphasizing the sense of tension and unease.\n \
        	4	Flecks of graphite from the pencil's tip may be visible, scattered across the black surface, as a result of the restless movements.\n \
        	5	The hand's knuckles and veins are accentuated by the lighting, highlighting the pressure and force exerted during the fidgeting.\n \
        	6	The pencil's eraser end, worn and discolored, suggests frequent use and a history of anxious behavior.\n \
        	7	A hint of perspiration on the hand's skin glistens under the light, further revealing the nervous energy within the scene.\n \
        	8	The positioning of the hand, perhaps slightly off-center or at an angle, contributes to the visual tension of the composition.\n \
        	9	Fingernails on the hand may appear bitten or worn, indicating a habit of nervousness and stress.\n \
        	10	The black background contrasts sharply with the hand and pencil, isolating them in the scene and focusing the viewer's attention on the restless, uneasy motion.\n \
                short: please describe what you might see in a picture of a scene that contains 'a man is programming', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to \n \
                long:   1	A focused man sits at a desk, his eyes intently scanning the computer screen in front of him as he works on a programming project.\n \
        	2	The computer display is filled with lines of code, featuring various colors and syntax highlighting to differentiate between elements of the programming language.\n \
        	3	The man's fingers move swiftly and confidently across the keyboard, typing commands and adjusting the code as needed.\n \
        	4	Beside the keyboard, a mouse and a notepad with handwritten notes or algorithms offer additional tools for the programmer's work.\n \
        	5	A cup of coffee or tea sits nearby, providing the man with a source of caffeine to maintain his focus and energy.\n \
        	6	The room's lighting, either from a desk lamp or overhead lights, illuminates the workspace, creating a comfortable environment for concentration.\n \
        	7	The man wears casual attire, such as a t-shirt and jeans, reflecting the informal nature of the programming process.\n \
        	8	Reference books or technical manuals may be stacked or spread out on the desk, offering guidance and information for the programmer.\n \
        	9	The man's facial expression, furrowed brows or a slight frown, conveys his deep concentration and determination to solve the coding challenge at hand.\n \
        	10	Surrounding the man, other electronic devices, like a smartphone or tablet, may be present, indicating the interconnected nature of his work in the digital realm.\n \
                short: please describe what you might see in a picture of a scene that contains '{short_description}', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to \n \
                long: "

        # Prepare input for the model
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
        # Generate multiple outputs and concatenate
        combined_output = ""
        num_generations = 3  # Number of outputs to generate
        
        # Generate text with mixed precision
        with autocast():
            for _ in range(num_generations):
                output = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=200,
                    min_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.7,
                    # top_k=100,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    attention_mask=inputs["attention_mask"],
                )

            # Decode the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Remove input prompt from generated output if necessary
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()
            generated_text = clean_output(generated_text)  # Clean the output

            combined_output += generated_text + "\n"

        # Filter hallucinations
        filtered_sentences = filter_hallucinations(short_description, combined_output, similarity_model, 0.5)
        print(short_description)
        print(filtered_sentences)
        # Write the filtered sentences to file
        with open(file_path, "w") as f:
            json.dump(filtered_sentences, f)
        torch.cuda.empty_cache()
            
if __name__ == "__main__":
    set_start_method("spawn")  # Use 'spawn' method for multiprocessing

    # Load models and tokenizer
    model_name = "EleutherAI/gpt-neo-2.7B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight similarity model

    # Process all JSON files
    json_files = glob.glob("../CC_10000/quality_captions/*.json")
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Perform pre-check for existing files
    unprocessed_files = []
    for json_file_name in json_files:
        clean_json_file_name = os.path.basename(json_file_name)[:-5]
        file_path = f"../CC_10000/LLM_ha_4/{clean_json_file_name}.json"
        if not os.path.isfile(file_path):
            unprocessed_files.append(json_file_name)

    print(f"Total files to process: {len(unprocessed_files)}")

    # Create tasks with GPU assignments
    tasks = [
        (json_file, "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-2.7B", "all-MiniLM-L6-v2", idx % num_gpus)
        for idx, json_file in enumerate(unprocessed_files)
    ]

    # Use multiprocessing to process files
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(process_file, *task)
            for task in tqdm(tasks, desc="Processing JSON files")
        ]
        for future in tqdm(futures, desc="Completing tasks"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")
