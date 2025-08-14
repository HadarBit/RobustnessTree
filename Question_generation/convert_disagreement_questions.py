#!/usr/bin/env python3
"""
Convert disagreement questions format to simple Q&A format for evaluation.
"""

import json
import sys

def convert_disagreement_questions(input_file, output_file):
    """Convert disagreement questions to simple Q&A format."""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract just the question and answer fields
    simple_questions = []
    
    for item in data["questions"]:
        simple_questions.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    # Save in simple format
    with open(output_file, 'w') as f:
        json.dump(simple_questions, f, indent=2)
    
    print(f"Converted {len(simple_questions)} questions")
    print(f"Original format: {input_file}")
    print(f"Simple format: {output_file}")
    
    # Show summary by capability
    capabilities = {}
    for item in data["questions"]:
        cap = item["capability"]
        if cap not in capabilities:
            capabilities[cap] = 0
        capabilities[cap] += 1
    
    print(f"\nQuestions by capability:")
    for cap, count in capabilities.items():
        print(f"  - {cap}: {count} questions")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_disagreement_questions.py <input_file> <output_file>")
        print("Example: python convert_disagreement_questions.py data/disagreement_questions_ranking_only.json data/simple_disagreement_questions.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_disagreement_questions(input_file, output_file) 