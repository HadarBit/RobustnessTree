#!/usr/bin/env python3
import json
import time
from together import Together
from dotenv import load_dotenv
import os

load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def evaluate_questions(questions_file):
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    success_vector = []
    detailed_results = []
    
    for i, q in enumerate(questions):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf",
                messages=[
                    {"role": "system", "content": "Answer with only the letter A, B, C, or D. No punctuation, no extra text, just the single letter."},
                    {"role": "user", "content": q["question"]}
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip().upper()
            # Extract just the letter, handle cases like "B.", "A. The answer", etc.
            predicted = 'X'
            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer:
                    predicted = letter
                    break
            success = 1 if predicted == q["answer"].upper() else 0
            
            result = {
                "question_number": i + 1,
                "question": q["question"],
                "correct_answer": q["answer"],
                "model_raw_output": response.choices[0].message.content,
                "parsed_answer": predicted,
                "success": success
            }
            
            success_vector.append(success)
            detailed_results.append(result)
            
            print(f"\n--- Question {i+1} ---")
            print(f"Question: {q['question'][:100]}...")
            print(f"Correct: {q['answer']}")
            print(f"Model raw output: '{response.choices[0].message.content}'")
            print(f"Parsed: {predicted}")
            print(f"Result: {'✓' if success else '✗'}")
            
            time.sleep(0.5)
            
        except Exception as e:
            result = {
                "question_number": i + 1,
                "question": q.get("question", ""),
                "correct_answer": q.get("answer", ""),
                "model_raw_output": "",
                "parsed_answer": "ERROR",
                "success": 0,
                "error": str(e)
            }
            
            success_vector.append(0)
            detailed_results.append(result)
            print(f"{i+1}: ERROR - {str(e)}")
    
    accuracy = sum(success_vector) / len(success_vector)
    
    # Create evaluation results
    evaluation_results = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "total_questions": len(questions),
        "correct_answers": sum(success_vector),
        "accuracy": accuracy,
        "success_vector": success_vector,
        "detailed_results": detailed_results
    }
    
    # Save to JSON file
    import os
    base_name = os.path.splitext(os.path.basename(questions_file))[0]
    output_file = f"evaluation_on_{base_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Vector: {success_vector}")
    print(f"Results saved to: {output_file}")
    
    return success_vector

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python EvaluationOnNewQuestions.py <questions.json>")
        sys.exit(1)
    
    evaluate_questions(sys.argv[1])