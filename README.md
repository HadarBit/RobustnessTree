# Robustness Tree Code

This toolkit processes **MMLU** questions, merges them with **DOVE robustness scores**, and generates **model weakness profiles** for deeper performance analysis.  
It is designed to work alongside the **EvalTree** framework, replacing accuracy-based rankings with robustness-based rankings.

---

## Python Files Flow

1. **`extract_mmlu_questions.py`**  
   Extracts questions from **MMLU** and produces a JSON file mapping each MMLU question to its index.

2. **`extract_dove_scores_Llama.py`**, **`extract_dove_scores_OLMoE.py`**  
   Extract **DOVE robustness scores** for **LLaMA** and **OLMoE** models.

3. **`merge_dove_score_with_mmlu_accuracy_score.py`**  
   Combines a model’s DOVE scores with the corresponding MMLU question indices.

4. **`replace_accuracy_ranking_in_DOVE.py`**  
   Updates **EvalTree** rankings by replacing accuracy scores with DOVE robustness scores.

5. **`weakness_question_generator.py`**  
   Produces a **weakness profile** summarizing model performance across different question types.

---

## Folder Structure

- **`EvalTree/`**  
  The original **EvalTree** repository (unmodified baseline).

- **`Replace Accuracy For DOVE Ranking/`**  
  Modified EvalTree trees where **accuracy scores** are replaced with **DOVE robustness scores**.

- **`internal_files/`**  
  Internal scripts and data used for generating plots.

- **`plots/`**  
  Generated visualizations illustrating the results.

- **`data/`**  
  Contains the **trees** and **tables** used in analysis, including input data for processing and evaluation.

---
