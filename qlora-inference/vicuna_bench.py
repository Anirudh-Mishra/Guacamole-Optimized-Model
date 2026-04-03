import json
import subprocess
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Full 80 Vicuna Benchmark Prompts ───
# Source: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/vicuna_bench/question.jsonl
VICUNA_PROMPTS = [
    {"id": 1, "category": "generic", "prompt": "How can I improve my time management skills?"},
    {"id": 2, "category": "generic", "prompt": "What are the most effective ways to deal with stress?"},
    {"id": 3, "category": "generic", "prompt": "What are the main differences between Python and JavaScript programming languages?"},
    {"id": 4, "category": "generic", "prompt": "How can I increase my productivity while working from home?"},
    {"id": 5, "category": "generic", "prompt": "Can you explain the basics of quantum computing?"},
    {"id": 6, "category": "generic", "prompt": "What are the differences between plant-based and animal-based protein sources?"},
    {"id": 7, "category": "generic", "prompt": "How can I develop my critical thinking skills?"},
    {"id": 8, "category": "generic", "prompt": "What are the major challenges faced by the education sector today?"},
    {"id": 9, "category": "generic", "prompt": "What are the primary factors that influence consumer behavior?"},
    {"id": 10, "category": "generic", "prompt": "What are the most effective strategies for conflict resolution in the workplace?"},
    {"id": 11, "category": "knowledge", "prompt": "What are some potential implications of using a single-use plastic bottle versus a reusable bottle on both the environment and human health?"},
    {"id": 12, "category": "knowledge", "prompt": "What factors would you consider when designing an inclusive and accessible public transportation system?"},
    {"id": 13, "category": "knowledge", "prompt": "How can governments utilize fiscal and monetary policies to combat economic recessions?"},
    {"id": 14, "category": "knowledge", "prompt": "How do language and cultural barriers affect the way people communicate and form relationships in multicultural societies?"},
    {"id": 15, "category": "knowledge", "prompt": "Describe a scenario where artificial intelligence could be used to improve the quality and efficiency of healthcare delivery."},
    {"id": 16, "category": "knowledge", "prompt": "Explain the process of gene editing using CRISPR-Cas9 technology, and discuss its potential applications and ethical implications."},
    {"id": 17, "category": "knowledge", "prompt": "How do vaccinations work to protect individuals and communities from infectious diseases, and what is herd immunity?"},
    {"id": 18, "category": "knowledge", "prompt": "How do social media platforms influence the way people consume and share news, and what are the potential implications for the spread of misinformation?"},
    {"id": 19, "category": "knowledge", "prompt": "How do cultural, social, and economic factors influence people's food choices, and how can this knowledge be used to promote healthier diets?"},
    {"id": 20, "category": "knowledge", "prompt": "Explain the process of natural selection and how it contributes to the evolution and adaptation of species."},
    {"id": 21, "category": "roleplay", "prompt": "How would you introduce yourself as a medieval knight at a royal banquet?"},
    {"id": 22, "category": "roleplay", "prompt": "As a pirate captain, what would you say to your crew to motivate them to search for hidden treasure?"},
    {"id": 23, "category": "roleplay", "prompt": "If you were a Shakespearean character, how would you declare your love for someone in a soliloquy?"},
    {"id": 24, "category": "roleplay", "prompt": "As a superhero, how would you explain your origin story to a curious child?"},
    {"id": 25, "category": "roleplay", "prompt": "Imagine you are a time traveler from the year 3000. What technological advancements would you tell people about?"},
    {"id": 26, "category": "roleplay", "prompt": "As a sports commentator, describe the winning play in the final seconds of a championship game."},
    {"id": 27, "category": "roleplay", "prompt": "Pretend to be a world-famous chef. How would you describe your signature dish to a panel of judges?"},
    {"id": 28, "category": "roleplay", "prompt": "You are a mountain climber reaching the summit of Mount Everest. Describe your emotions and the view from the top."},
    {"id": 29, "category": "roleplay", "prompt": "As a space colonist on Mars, describe your daily life and the challenges you face living on another planet."},
    {"id": 30, "category": "roleplay", "prompt": "Pretend to be a character in a post-apocalyptic world. Describe how you survive and the allies you encounter."},
    {"id": 31, "category": "common-sense", "prompt": "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?"},
    {"id": 32, "category": "common-sense", "prompt": "What are some subtle clues that suggest someone is pretending to understand a topic or conversation when they are actually confused or uninformed?"},
    {"id": 33, "category": "common-sense", "prompt": "Why might someone choose to use a paper map or ask for directions instead of relying on a GPS device or smartphone app?"},
    {"id": 34, "category": "common-sense", "prompt": "How can you determine if a person is genuinely interested in a conversation or simply being polite?"},
    {"id": 35, "category": "common-sense", "prompt": "Why might someone prefer to shop at a small, locally-owned business instead of a large chain store, even if the prices are higher?"},
    {"id": 36, "category": "common-sense", "prompt": "How can you assess the credibility of a source of information, such as a news article or blog post, without relying solely on the reputation of the author or publisher?"},
    {"id": 37, "category": "common-sense", "prompt": "Why do some people enjoy the sensation of being scared, such as by watching horror movies or going on roller coasters, while others avoid these experiences?"},
    {"id": 38, "category": "common-sense", "prompt": "How can observing the behavior of other people in a social situation provide clues about cultural norms and expectations?"},
    {"id": 39, "category": "common-sense", "prompt": "Do we have a moral obligation to explore space, or should we focus on solving Earth's problems first?"},
    {"id": 40, "category": "common-sense", "prompt": "In a world where automation is becoming increasingly prevalent, is it more important to prioritize job creation or technological progress?"},
    {"id": 41, "category": "fermi", "prompt": "How many times does the average human blink in a lifetime? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 42, "category": "fermi", "prompt": "How many atoms are in a grain of salt? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 43, "category": "fermi", "prompt": "How many lightning strikes occur on Earth each day? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 44, "category": "fermi", "prompt": "How many balloons would it take to lift a house like in the movie Up? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 45, "category": "fermi", "prompt": "How many text messages are sent globally in a minute? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 46, "category": "fermi", "prompt": "How many words are spoken daily on Earth? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 47, "category": "fermi", "prompt": "How many snowflakes fall during a typical winter? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 48, "category": "fermi", "prompt": "How many pages are in all the books ever written? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 49, "category": "fermi", "prompt": "How many times has the Earth orbited the Sun since the beginning of life? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 50, "category": "fermi", "prompt": "How many songs have been recorded throughout history? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."},
    {"id": 51, "category": "counterfactual", "prompt": "What if the Internet had been invented during the Renaissance period?"},
    {"id": 52, "category": "counterfactual", "prompt": "What if the Aztecs had successfully repelled the Spanish conquistadors?"},
    {"id": 53, "category": "counterfactual", "prompt": "What if the Black Death had not occurred in the 14th century?"},
    {"id": 54, "category": "counterfactual", "prompt": "What if Isaac Newton had focused on biology instead of physics?"},
    {"id": 55, "category": "counterfactual", "prompt": "What if the Beatles had never formed as a band?"},
    {"id": 56, "category": "counterfactual", "prompt": "What if Alan Turing had not cracked the Enigma code during World War II?"},
    {"id": 57, "category": "counterfactual", "prompt": "What if the Suez Canal had never been constructed?"},
    {"id": 58, "category": "counterfactual", "prompt": "What if the Maya civilization had never mysteriously collapsed?"},
    {"id": 59, "category": "counterfactual", "prompt": "What if Christopher Columbus had not discovered the Americas?"},
    {"id": 60, "category": "counterfactual", "prompt": "What if Vincent van Gogh had been a successful artist during his lifetime?"},
    {"id": 61, "category": "coding", "prompt": "Develop a C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file."},
    {"id": 62, "category": "coding", "prompt": "Implement a Python function to find the longest common subsequence of two input strings using dynamic programming."},
    {"id": 63, "category": "coding", "prompt": "Implement a regular expression in Python to validate an email address."},
    {"id": 64, "category": "coding", "prompt": "Write a program to find the nth Fibonacci number using dynamic programming."},
    {"id": 65, "category": "coding", "prompt": "Implement a binary search algorithm to find a specific element in a sorted array."},
    {"id": 66, "category": "coding", "prompt": "Implement a queue data structure using two stacks in Python."},
    {"id": 67, "category": "coding", "prompt": "Implement a program to find the common elements in two arrays without using any extra data structures."},
    {"id": 68, "category": "math", "prompt": "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2)."},
    {"id": 69, "category": "math", "prompt": "Solve for x in the equation 3x + 10 = 5(x - 2)."},
    {"id": 70, "category": "math", "prompt": "If the endpoints of a line segment are (2, -2) and (10, 4), what is the length of the segment?"},
    {"id": 71, "category": "writing", "prompt": "Can you help me write a formal email to a potential business partner proposing a joint venture?"},
    {"id": 72, "category": "writing", "prompt": "Can you help me write a resignation letter to my current employer, while leaving on good terms and expressing gratitude for the opportunities provided?"},
    {"id": 73, "category": "writing", "prompt": "Use an appropriate format to structure a formal letter of recommendation for a student applying to a prestigious graduate program in computer science."},
    {"id": 74, "category": "writing", "prompt": "Write a compelling product launch announcement email to inform our customers of our new software solution."},
    {"id": 75, "category": "writing", "prompt": "Draft an apology email to a customer who experienced a delay in their order, and provide reassurance that the issue has been resolved."},
    {"id": 76, "category": "writing", "prompt": "Write a script for a YouTube video exploring the history and cultural significance of jazz."},
    {"id": 77, "category": "writing", "prompt": "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."},
    {"id": 78, "category": "writing", "prompt": "Write a captivating movie review for a recently released science fiction film, discussing its plot, characters, and special effects."},
    {"id": 79, "category": "writing", "prompt": "Structure a podcast script for an episode discussing the influence of streaming platforms on the music industry."},
    {"id": 80, "category": "writing", "prompt": "Write a symphony concert review, discussing the orchestra's performance and overall audience experience."},
]

print(f"Loaded {len(VICUNA_PROMPTS)} Vicuna benchmark prompts across {len(set(p['category'] for p in VICUNA_PROMPTS))} categories")

# ─── Generate response from our model via MLX ───
def generate_response(prompt, model_path, adapter_path):
    formatted = f"### Human: {prompt}\n### Assistant:"
    try:
        result = subprocess.run(
            [
                "python", "-m", "mlx_lm", "generate",
                "--model", model_path,
                "--adapter-path", adapter_path,
                "--prompt", formatted,
                "--max-tokens", "500",
            ],
            capture_output=True, text=True, timeout=180
        )
        output = result.stdout
        parts = output.split("==========")
        if len(parts) >= 2:
            response = parts[1].strip()
            if "### Human:" in response:
                response = response.split("### Human:")[0].strip()
            return response
        return output.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"

# ─── Score with GPT-4o-mini ───
def score_with_gpt4(prompt, model_response, client):
    scoring_prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{prompt}

[The Start of Assistant's Answer]
{model_response}
[The End of Assistant's Answer]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": scoring_prompt}],
            max_tokens=500,
            temperature=0,
        )
        reply = response.choices[0].message.content
        match = re.search(r'\[\[(\d+)\]\]', reply)
        rating = int(match.group(1)) if match else None
        return {"rating": rating, "explanation": reply}
    except Exception as e:
        return {"rating": None, "explanation": f"[ERROR: {str(e)}]"}

# ─── Main ───
def main():
    MODEL_PATH = "./inference/mlx-llama3-8b"
    ADAPTER_PATH = "./adapter-mlx"
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=sk-...")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Resume from previous run if exists
    results = []
    completed_ids = set()
    if os.path.exists("vicuna_results.json"):
        with open("vicuna_results.json", "r") as f:
            results = json.load(f)
            completed_ids = {r["question_id"] for r in results}
            print(f"Resuming: {len(completed_ids)}/80 prompts already completed")
    
    for p in VICUNA_PROMPTS:
        if p["id"] in completed_ids:
            continue
        
        print(f"\n{'='*60}")
        print(f"[{p['id']}/80] [{p['category']}] {p['prompt'][:70]}...")
        print(f"{'='*60}")
        
        print("  Generating response...")
        model_response = generate_response(p["prompt"], MODEL_PATH, ADAPTER_PATH)
        print(f"  Response: {model_response[:150]}...")
        
        print("  Scoring with GPT-4o-mini...")
        score = score_with_gpt4(p["prompt"], model_response, client)
        print(f"  Rating: {score['rating']}/10")
        
        results.append({
            "question_id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "model_response": model_response,
            "rating": score["rating"],
            "explanation": score["explanation"],
        })
        
        with open("vicuna_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # ─── Summary ───
    ratings = [r["rating"] for r in results if r["rating"] is not None]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    
    print(f"\n{'='*60}")
    print(f"VICUNA BENCHMARK RESULTS (Full 80 prompts)")
    print(f"{'='*60}")
    print(f"Prompts evaluated: {len(ratings)}/80")
    print(f"Overall average: {avg_rating:.2f}/10")
    print(f"Min: {min(ratings)}/10 | Max: {max(ratings)}/10")
    
    categories = sorted(set(r["category"] for r in results))
    print(f"\nPer-category breakdown:")
    for cat in categories:
        cat_ratings = [r["rating"] for r in results if r["category"] == cat and r["rating"] is not None]
        if cat_ratings:
            avg = sum(cat_ratings) / len(cat_ratings)
            print(f"  {cat:20s}: {avg:.2f}/10  (n={len(cat_ratings)})")
    
    print(f"\nAll results saved to vicuna_results.json")

if __name__ == "__main__":
    main()