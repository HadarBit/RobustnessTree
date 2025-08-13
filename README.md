MMLU–DOVE Integration Toolkit
This toolkit processes MMLU questions, merges them with DOVE robustness scores, and generates model weakness profiles for deeper performance analysis.

Python Files Flow
extract_mmlu_questions.py
Extracts questions from MMLU and creates a JSON file mapping MMLU questions to their indices.

extract_dove_scores_Llama.py, extract_dove_scores_OLMoE.py
Extracts DOVE scores for LLaMA and OLMoE models.

merge_dove_score_with_mmlu_accuracy_score.py
Merges a model’s DOVE score with the corresponding MMLU question index.

replace_accuracy_ranking_in_DOVE.py
Replaces the accuracy score in EvalTree with the DOVE robustness score.

weakness_question_generator.py
Generates a weakness profile based on the integrated scores.
