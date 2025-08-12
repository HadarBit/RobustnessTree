import json
mmlu_dove_score_path = "data/MMLU_DOVE.json"
dove_question_to_index = "data/mmlu_question_subject.json"

with open(mmlu_dove_score_path, 'r') as f:
    dove_scores = json.load(f)


with open(dove_question_to_index, 'r') as f:
    idx_to_question = json.load(f)

flatten_dict = {}
for category  in idx_to_question.keys():
    flatten_dict.update(idx_to_question[category])
    
question_to_dove_score = {}
for idx in flatten_dict.keys():
    text_question = flatten_dict[idx]
    if idx in dove_scores:
        dove_score = dove_scores[idx]
        question_to_dove_score[text_question] = dove_score

with open("./data/dove_question_to_dove_score.json", "w") as f:
    json.dump(question_to_dove_score, f, indent=2)

    