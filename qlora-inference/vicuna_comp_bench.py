import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

"""

It sends the same 80 prompts to ChatGPT (GPT-3.5 Turbo),
scores ChatGPT's responses with the same GPT-4o-mini judge,
and computes the ratio: (our model score) / (ChatGPT score) x 100%
This replicates the QLoRA paper's Vicuna evaluation methodology.
"""

# ─── Score with GPT-4o-mini (same function as vicuna_bench.py) ───
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

# ─── Generate ChatGPT response ───
def get_chatgpt_response(prompt, client):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {str(e)}]"

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=sk-...")
        return

    # Load our model's results
    if not os.path.exists("vicuna_results.json"):
        print("ERROR: vicuna_results.json not found. Run vicuna_bench.py first.")
        return

    with open("vicuna_results.json", "r") as f:
        our_results = json.load(f)

    print(f"Loaded {len(our_results)} results from our model")

    client = OpenAI(api_key=api_key)

    # Resume from previous run if exists
    chatgpt_results = []
    completed_ids = set()
    if os.path.exists("chatgpt_results.json"):
        with open("chatgpt_results.json", "r") as f:
            chatgpt_results = json.load(f)
            completed_ids = {r["question_id"] for r in chatgpt_results}
            print(f"Resuming: {len(completed_ids)}/80 ChatGPT responses already completed")

    for r in our_results:
        if r["question_id"] in completed_ids:
            continue

        print(f"\n{'='*60}")
        print(f"[{r['question_id']}/80] [{r['category']}] {r['prompt'][:70]}...")
        print(f"{'='*60}")

        # Get ChatGPT response
        print("  Getting ChatGPT response...")
        chatgpt_response = get_chatgpt_response(r["prompt"], client)
        print(f"  ChatGPT: {chatgpt_response[:150]}...")

        # Score ChatGPT response with same judge
        print("  Scoring ChatGPT with GPT-4o-mini...")
        score = score_with_gpt4(r["prompt"], chatgpt_response, client)
        print(f"  ChatGPT rating: {score['rating']}/10 (Our model: {r['rating']}/10)")

        chatgpt_results.append({
            "question_id": r["question_id"],
            "category": r["category"],
            "prompt": r["prompt"],
            "chatgpt_response": chatgpt_response,
            "chatgpt_rating": score["rating"],
            "chatgpt_explanation": score["explanation"],
            "our_rating": r["rating"],
        })

        # Save incrementally
        with open("chatgpt_results.json", "w") as f:
            json.dump(chatgpt_results, f, indent=2)

    # ─── Final Comparison ───
    print(f"\n{'='*60}")
    print(f"VICUNA BENCHMARK: OUR MODEL vs CHATGPT")
    print(f"{'='*60}")

    # Merge results
    our_ratings = {r["question_id"]: r["rating"] for r in our_results if r["rating"] is not None}
    gpt_ratings = {r["question_id"]: r["chatgpt_rating"] for r in chatgpt_results if r["chatgpt_rating"] is not None}

    # Only compare questions where both have scores
    common_ids = set(our_ratings.keys()) & set(gpt_ratings.keys())
    
    our_scores = [our_ratings[qid] for qid in common_ids]
    gpt_scores = [gpt_ratings[qid] for qid in common_ids]

    our_avg = sum(our_scores) / len(our_scores)
    gpt_avg = sum(gpt_scores) / len(gpt_scores)
    ratio = (our_avg / gpt_avg) * 100

    print(f"\nQuestions compared: {len(common_ids)}/80")
    print(f"Our model average:  {our_avg:.2f}/10")
    print(f"ChatGPT average:    {gpt_avg:.2f}/10")
    print(f"")
    print(f">>> Our model achieves {ratio:.1f}% of ChatGPT's performance <<<")
    print(f"")
    print(f"QLoRA paper reference: Guanaco-7B = 87.0% of ChatGPT")

    # Per-category comparison
    categories = sorted(set(r["category"] for r in chatgpt_results))
    print(f"\nPer-category breakdown:")
    print(f"  {'Category':20s} {'Ours':>8s} {'ChatGPT':>8s} {'Ratio':>8s}")
    print(f"  {'-'*48}")
    for cat in categories:
        cat_our = [our_ratings[r["question_id"]] for r in chatgpt_results
                   if r["category"] == cat and r["question_id"] in our_ratings and r["chatgpt_rating"] is not None]
        cat_gpt = [r["chatgpt_rating"] for r in chatgpt_results
                   if r["category"] == cat and r["chatgpt_rating"] is not None and r["question_id"] in our_ratings]
        if cat_our and cat_gpt:
            our_cat_avg = sum(cat_our) / len(cat_our)
            gpt_cat_avg = sum(cat_gpt) / len(cat_gpt)
            cat_ratio = (our_cat_avg / gpt_cat_avg) * 100 if gpt_cat_avg > 0 else 0
            print(f"  {cat:20s} {our_cat_avg:>7.2f} {gpt_cat_avg:>8.2f} {cat_ratio:>7.1f}%")

    # Save final comparison
    comparison = {
        "questions_compared": len(common_ids),
        "our_model_average": round(our_avg, 2),
        "chatgpt_average": round(gpt_avg, 2),
        "performance_ratio": round(ratio, 1),
        "paper_reference": "Guanaco-7B = 87.0% of ChatGPT",
        "judge_model": "gpt-4o-mini",
        "chatgpt_model": "gpt-3.5-turbo",
    }
    with open("vicuna_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to vicuna_comparison.json")

if __name__ == "__main__":
    main()