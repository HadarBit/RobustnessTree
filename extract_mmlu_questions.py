import json
import os

from datasets import load_dataset
from tqdm import tqdm


def create_mmlu_question_index_mapping():
    """
    Create a JSON file mapping MMLU questions to their indices.
    Format: {"question_text": index, "question_text2": index2, ...}
    """

    print("Loading CAIS/MMLU dataset...")

    # Load the dataset
    try:
        dataset = load_dataset("cais/mmlu", "all")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        dataset = load_dataset("cais/mmlu")

    # Dictionary to store question -> index mapping
    question_to_index = {}

    # Process all splits (auxiliary_train, dev, test, val)
    splits_to_process = []
    for split_name in dataset.keys():
        if dataset[split_name]:  # Check if split is not empty
            splits_to_process.append(split_name)

    print(f"Processing splits: {splits_to_process}")

    global_index = 0

    for split_name in splits_to_process:
        split_data = dataset[split_name]
        print(f"\nProcessing {split_name} split with {len(split_data)} examples...")

        for local_index, example in enumerate(
            tqdm(split_data, desc=f"Processing {split_name}")
        ):
            question_text = example["question"]

            # Handle potential duplicate questions by adding split info if needed
            if question_text in question_to_index:
                # If question already exists, create a unique key
                unique_key = (
                    f"{question_text} [split: {split_name}, local_idx: {local_index}]"
                )
                question_to_index[unique_key] = global_index
            else:
                question_to_index[question_text] = global_index

            global_index += 1

    print(f"\nTotal questions processed: {len(question_to_index)}")

    # Save to JSON file
    output_file = "mmlu_question_to_index.json"
    print(f"Saving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(question_to_index, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

    # Display some sample mappings
    print("\nSample mappings (first 5):")
    sample_items = list(question_to_index.items())[:5]
    for question, index in sample_items:
        print(f"Index {index}: {question[:100]}{'...' if len(question) > 100 else ''}")

    return question_to_index


def create_subject_wise_mapping():
    """
    Alternative function to create subject-wise question mappings.
    This creates separate mappings for each subject.
    """
    print("Loading CAIS/MMLU dataset for subject-wise mapping...")

    # MMLU subjects
    subjects = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    all_questions = {}
    global_index = 0

    for subject in tqdm(subjects, desc="Processing subjects"):
        try:
            subject_dataset = load_dataset("cais/mmlu", subject)

            for split_name in ["test", "dev", "val"]:  # Common splits
                if split_name in subject_dataset:
                    split_data = subject_dataset[split_name]

                    for example in split_data:
                        question_text = example["question"]
                        # Add subject context to avoid duplicates across subjects
                        key = f"{question_text} [subject: {subject}]"
                        all_questions[key] = global_index
                        global_index += 1

        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            continue

    # Save subject-wise mapping
    output_file = "mmlu_subject_wise_question_to_index.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Subject-wise mapping saved to {output_file}")
    print(f"Total questions: {len(all_questions)}")

    return all_questions


if __name__ == "__main__":
    print("MMLU Question to Index Mapper")
    print("=" * 40)

    choice = input(
        "Choose mapping type:\n1. All splits combined\n2. Subject-wise mapping\nEnter choice (1 or 2): "
    )

    if choice == "1":
        question_to_index = create_mmlu_question_index_mapping()
    elif choice == "2":
        question_to_index = create_subject_wise_mapping()
    else:
        print("Invalid choice. Running default (all splits combined)...")
        question_to_index = create_mmlu_question_index_mapping()

    print("\nDone! You can now use the generated JSON file.")
