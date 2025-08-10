Read the MMLU.json and understand its strucutre its a herrarchical evaluation of MMLU where ranking is scores of models on subtrees. When the size is 1 meaning its a leaf node i want you to add a new ranking called "dove_ranking" remove all other ranking but the "Llama-3.1-8B-Instruct". this is current leaf ranking values "ranking": [
[
"gpt-4o-mini-2024-07-18",
1.0
],
[
"gpt-3.5-turbo",
1.0
],
[
"claude-3.5-haiku",
1.0
],

                                    [
                                      "Llama-3.1-Tulu-3-70B",
                                      1.0
                                    ],
                                    [
                                      "Qwen2.5-7B-Instruct",
                                      1.0
                                    ],
                                    [
                                      "Qwen2.5-72B-Instruct",
                                      1.0
                                    ],
                                    [
                                      "Llama-3.1-Tulu-3-8B",
                                      0.0
                                    ]
                                  ],
                                  "distinction": "Client relationship risk evaluation"

we want after this code runs that the leaf will have "ranking": only llama3.18b and "dove ranking" in the same herrarchy as rnaking and its value is via question as key which you find in 'input' in the MMLU.json and the evaluation of that quesiton is inside the dove_question_to_dove_score.json where key is the question text and valuie is the result.
