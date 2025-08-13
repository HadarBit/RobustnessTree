#!/usr/bin/env python3
"""
MMLU Question Generation Prompts

Contains prompts for generating MMLU-style questions targeting specific capabilities.
"""

def get_mmlu_generation_prompt(capability: str, example_questions: list, num_examples: int = 3) -> tuple:
    """
    Generate system and user prompts for MMLU-style question generation.
    
    Args:
        capability: The capability to test
        example_questions: List of example questions for reference
        num_examples: Number of examples being provided
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    
    system_prompt = """You are an expert question writer creating MMLU-style multiple-choice questions for academic assessment."""

    user_prompt = f"""Generate one CHALLENGING MMLU-style multiple choice question with exactly 4 options (A, B, C, D) testing this capability:

**Capability:** {capability}

**Requirements:**
- Create a DIFFICULT question that requires deep understanding
- Standard MMLU format with 4 labeled options
- Only ONE correct answer
- Make incorrect options very plausible and tempting
- Require multi-step reasoning or advanced knowledge
- Original content (do not copy examples)

**Example Questions (these represent areas where models typically struggle):**
{chr(10).join([f"{i+1}. {ex[:200]}..." if len(ex) > 200 else f"{i+1}. {ex}" for i, ex in enumerate(example_questions[:num_examples])])}

**Make your question MORE challenging than these examples by:**
- Adding complexity or multiple concepts
- Using advanced terminology or subtle distinctions
- Requiring deeper analytical thinking
- Including common misconceptions as wrong answers

**Output Format:**
[Question text here]

A. [Option]
B. [Option] 
C. [Option]
D. [Option]

IMPORTANT: Randomly choose which option (A, B, C, or D) is correct. Mix it up - don't always make A correct. Generate only the question - no explanations or additional text."""

    return system_prompt, user_prompt 