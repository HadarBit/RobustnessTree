#!/usr/bin/env python3
"""
Weakness-Based Question Generator and Evaluator

This system generates new questions targeting identified weakness areas and evaluates
them to validate that our weakness profiling captures real model vulnerabilities.
"""

import json
import random
import sys
import time
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv
from together import Together


class WeaknessQuestionGenerator:
    """Generates questions targeting specific weakness capabilities."""

    def __init__(self, openai_api_key: str = None, together_api_key: str = None):
        """Initialize with API keys."""
        # Use provided keys or fall back to hardcoded ones
        self.openai_api_key = openai_api_key
        self.together_api_key = ()

        # Set up API clients
        if self.together_api_key:
            self.together_client = Together(api_key=self.together_api_key)
        else:
            self.together_client = None

    def load_weakness_profile(self, profile_path: str) -> Dict[str, Any]:
        """Load the weakness profile data."""
        with open(profile_path, "r") as f:
            return json.load(f)

    def extract_example_questions(
        self, weakness_node: Dict[str, Any], num_examples: int = 3
    ) -> List[str]:
        """Extract example questions from a weakness node."""
        examples = []

        def extract_from_node(node):
            if isinstance(node.get("subtrees"), (int, type(None))) and "input" in node:
                # Clean up the question text
                question_text = node["input"]
                if question_text.startswith("Question: "):
                    question_text = question_text[10:]  # Remove "Question: " prefix
                examples.append(question_text.strip())
            elif isinstance(node.get("subtrees"), list):
                for child in node["subtrees"]:
                    extract_from_node(child)

        extract_from_node(weakness_node)

        # Return a random sample of examples
        return random.sample(examples, min(num_examples, len(examples)))

    def generate_question_prompt(
        self, capability: str, example_questions: List[str], num_examples: int = 3
    ) -> str:
        """Generate the prompt for question generation."""

        system_prompt = """You are a creative and logical assistant tasked with generating new mathematics questions. Your goal is to create a single, clear question aligned with a given mathematical capability."""

        user_prompt = f"""## Task
Generate one unique mathematics question MCQ with 4 options demonstrating the following capability:
{capability}

Please ensure the following:
- You will be given {num_examples} example questions for reference. Use the examples solely to understand the capability, NOT as templates, i.e., the generated question must not replicate, paraphrase, or directly resemble the example questions in structure, wording, or context.
- The question must ask for only one result, such as a numerical value, while adhering to logical constraints (e.g., quantities must be positive, and counts for people must be integers).

## Provided Examples
{chr(10).join([f"{i+1}. {ex}" for i, ex in enumerate(example_questions)])}

## Requirements
- Do NOT include a solution in the generated question.
- Ensure the question is plausible, reasonable, and relevant to the given capability.
- Generate ONLY the question text, nothing else."""

        return system_prompt, user_prompt

    def generate_question_via_openai(
        self, capability: str, example_questions: List[str]
    ) -> str:
        """Generate a question using GPT-4o-mini."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")

        system_prompt, user_prompt = self.generate_question_prompt(
            capability, example_questions
        )

        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.8,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating question with OpenAI: {e}")
            return None

    def evaluate_question_via_together(
        self, question: str, model_name: str = "meta-llama/Llama-3-8b-chat-hf"
    ) -> Dict[str, Any]:
        """Evaluate a question using Together AI."""
        if not self.together_client:
            raise ValueError("Together AI client not initialized")

        try:
            response = self.together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please answer this question: {question}",
                    }
                ],
                max_tokens=300,
                temperature=0.1,  # Low temperature for consistent evaluation
            )

            answer = response.choices[0].message.content.strip()

            return {
                "question": question,
                "model_answer": answer,
                "model": model_name,
                "success": True,
            }

        except Exception as e:
            print(f"Error evaluating question with Together AI: {e}")
            return {
                "question": question,
                "model_answer": None,
                "model": model_name,
                "success": False,
                "error": str(e),
            }

    def generate_weakness_questions(
        self, profile_path: str, num_questions_per_weakness: int = 5
    ) -> Dict[str, Any]:
        """Generate questions for all weakness areas."""

        print("Loading weakness profile...")
        profile = self.load_weakness_profile(profile_path)

        generated_questions = {
            "metadata": {
                "model_name": profile["model_name"],
                "extraction_params": profile["extraction_params"],
                "generation_timestamp": time.time(),
                "num_questions_per_weakness": num_questions_per_weakness,
            },
            "weakness_questions": [],
        }

        print(
            f"Generating questions for {len(profile['weakness_nodes'])} weakness areas..."
        )

        for i, weakness in enumerate(profile["weakness_nodes"]):
            print(f"\nProcessing weakness {i+1}: {weakness['capability'][:60]}...")

            # Extract example questions
            example_questions = self.extract_example_questions(weakness["node_data"])
            print(f"  Found {len(example_questions)} example questions")

            weakness_questions = {
                "weakness_id": i,
                "capability": weakness["capability"],
                "original_size": weakness["size"],
                "original_accuracy": weakness["accuracy"],
                "original_dove_avg": sum(weakness["dove_scores"])
                / len(weakness["dove_scores"]),
                "subjects": weakness["subjects"],
                "example_questions": example_questions,
                "generated_questions": [],
            }

            # Generate multiple questions for this weakness
            for q_num in range(num_questions_per_weakness):
                print(
                    f"    Generating question {q_num + 1}/{num_questions_per_weakness}..."
                )

                generated_q = self.generate_question_via_openai(
                    weakness["capability"], example_questions
                )

                if generated_q:
                    weakness_questions["generated_questions"].append(
                        {
                            "question_id": q_num,
                            "question_text": generated_q,
                            "generated_successfully": True,
                        }
                    )
                    print(f"      ✓ Generated: {generated_q[:80]}...")
                else:
                    weakness_questions["generated_questions"].append(
                        {
                            "question_id": q_num,
                            "question_text": None,
                            "generated_successfully": False,
                        }
                    )
                    print(f"      ✗ Failed to generate question {q_num + 1}")

                # Small delay to respect API limits
                time.sleep(1)

            generated_questions["weakness_questions"].append(weakness_questions)

        return generated_questions

    def evaluate_generated_questions(
        self, generated_questions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate all generated questions using Together AI."""

        print("Evaluating generated questions...")

        evaluation_results = {
            "metadata": generated_questions["metadata"],
            "evaluation_results": [],
        }

        for weakness_data in generated_questions["weakness_questions"]:
            print(f"\nEvaluating questions for: {weakness_data['capability'][:60]}...")

            weakness_results = {
                "weakness_id": weakness_data["weakness_id"],
                "capability": weakness_data["capability"],
                "original_accuracy": weakness_data["original_accuracy"],
                "question_evaluations": [],
            }

            for q_data in weakness_data["generated_questions"]:
                if q_data["generated_successfully"]:
                    print(f"  Evaluating: {q_data['question_text'][:60]}...")

                    eval_result = self.evaluate_question_via_together(
                        q_data["question_text"]
                    )

                    weakness_results["question_evaluations"].append(
                        {
                            "question_id": q_data["question_id"],
                            "question_text": q_data["question_text"],
                            "evaluation": eval_result,
                        }
                    )

                    # Small delay to respect API limits
                    time.sleep(2)
                else:
                    weakness_results["question_evaluations"].append(
                        {
                            "question_id": q_data["question_id"],
                            "question_text": None,
                            "evaluation": {
                                "success": False,
                                "error": "Question generation failed",
                            },
                        }
                    )

            evaluation_results["evaluation_results"].append(weakness_results)

        return evaluation_results

    def analyze_validation_results(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the evaluation results to validate weakness predictions."""

        print("\n" + "=" * 80)
        print("WEAKNESS VALIDATION ANALYSIS")
        print("=" * 80)

        analysis = {
            "summary": {
                "total_weaknesses": len(evaluation_results["evaluation_results"]),
                "total_questions_generated": 0,
                "total_questions_evaluated": 0,
                "validation_success": True,
            },
            "weakness_analysis": [],
        }

        for weakness_result in evaluation_results["evaluation_results"]:
            capability = weakness_result["capability"]
            original_accuracy = weakness_result["original_accuracy"]

            print(f"\n📊 WEAKNESS: {capability[:60]}...")
            print(f"   Original accuracy: {original_accuracy:.3f}")

            evaluations = weakness_result["question_evaluations"]
            successful_evals = [e for e in evaluations if e["evaluation"]["success"]]

            analysis["summary"]["total_questions_generated"] += len(evaluations)
            analysis["summary"]["total_questions_evaluated"] += len(successful_evals)

            weakness_analysis = {
                "capability": capability,
                "original_accuracy": original_accuracy,
                "questions_generated": len(evaluations),
                "questions_evaluated": len(successful_evals),
                "model_answers": [
                    e["evaluation"]["model_answer"] for e in successful_evals
                ],
            }

            print(f"   Generated questions: {len(evaluations)}")
            print(f"   Successfully evaluated: {len(successful_evals)}")

            if successful_evals:
                print(f"   Sample answers:")
                for i, eval_data in enumerate(successful_evals[:2]):  # Show first 2
                    answer = eval_data["evaluation"]["model_answer"]
                    print(f"     Q{i+1}: {answer[:100]}...")

            analysis["weakness_analysis"].append(weakness_analysis)

        print(f"\n🎯 OVERALL VALIDATION RESULTS:")
        print(f"   Total weaknesses tested: {analysis['summary']['total_weaknesses']}")
        print(
            f"   Total questions generated: {analysis['summary']['total_questions_generated']}"
        )
        print(
            f"   Total questions evaluated: {analysis['summary']['total_questions_evaluated']}"
        )

        success_rate = (
            analysis["summary"]["total_questions_evaluated"]
            / analysis["summary"]["total_questions_generated"]
            if analysis["summary"]["total_questions_generated"] > 0
            else 0
        )
        print(f"   Evaluation success rate: {success_rate:.1%}")

        return analysis


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and evaluate questions for weakness validation"
    )
    parser.add_argument(
        "--profile", required=True, help="Path to weakness profile JSON"
    )
    parser.add_argument("--output", required=True, help="Output file prefix")
    parser.add_argument(
        "--num-questions", type=int, default=3, help="Number of questions per weakness"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate questions, don't evaluate",
    )

    args = parser.parse_args()
    load_dotenv()
    # Get API keys from args or environment
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    load_dotenv()
    if not openai_key:
        print("❌ OpenAI API key required for question generation!")
        print("   Set --openai-key or OPENAI_API_KEY environment variable")
        sys.exit(1)

    if not args.generate_only and not together_key:
        print("❌ Together AI API key required for evaluation!")
        print("   Set --together-key or TOGETHER_API_KEY environment variable")
        print("   Or use --generate-only to skip evaluation")
        sys.exit(1)

    # Initialize generator
    generator = WeaknessQuestionGenerator(openai_key, together_key)

    # Generate questions
    print("🚀 Starting weakness-based question generation...")
    generated_questions = generator.generate_weakness_questions(
        args.profile, args.num_questions
    )

    # Save generated questions
    questions_file = f"{args.output}_generated_questions.json"
    with open(questions_file, "w") as f:
        json.dump(generated_questions, f, indent=2)
    print(f"💾 Generated questions saved to {questions_file}")

    if not args.generate_only:
        # Evaluate questions
        print("\n🔍 Starting question evaluation...")
        evaluation_results = generator.evaluate_generated_questions(generated_questions)

        # Save evaluation results
        eval_file = f"{args.output}_evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"💾 Evaluation results saved to {eval_file}")

        # Analyze results
        analysis = generator.analyze_validation_results(evaluation_results)

        # Save analysis
        analysis_file = f"{args.output}_validation_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"💾 Validation analysis saved to {analysis_file}")

        print(f"\n✅ Complete! Files created:")
        print(f"   - {questions_file}")
        print(f"   - {eval_file}")
        print(f"   - {analysis_file}")
    else:
        print(f"\n✅ Question generation complete! File created:")
        print(f"   - {questions_file}")
        print(f"   Use the evaluation step separately with your Together AI key.")


if __name__ == "__main__":
    main()
