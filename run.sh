# python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path ./predictions/subtaskA_mono_xlnet.jsonl --subtask A --model xlnet-base-cased

#python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path ./predictions/subtaskA_mono_roberta.jsonl --subtask A --model roberta-base

#python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_multilingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_multilingual.jsonl --prediction_file_path ./predictions/subtaskA_mul_roberta.jsonl --subtask A --model roberta-base


#python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path ./predictions/subtaskA_mono_xlmr.jsonl --subtask A --model xlm-roberta-base

#python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_multilingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_multilingual.jsonl --prediction_file_path ./predictions/subtaskA_mul_xlmr.jsonl --subtask A --model xlm-roberta-base


#python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/subtaskA/SubtaskA_train_monolingual.jsonl --test_file_path ./data/subtaskA/SubtaskA_dev_monolingual.jsonl --prediction_file_path ./predictions/subtaskA_mono_mistral.jsonl --subtask A --model mistralai/Mistral-7B-v0.1

#python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/subtaskA/SubtaskA_train_multilingual.jsonl --test_file_path ./data/subtaskA/SubtaskA_dev_multilingual.jsonl --prediction_file_path ./predictions/subtaskA_mul_mistral.jsonl --subtask A --model mistralai/Mistral-7B-v0.1


#python3 subtaskB/baseline/transformer_baseline.py --train_file_path ./data/SubtaskB/subtaskB_train.jsonl --test_file_path ./data/SubtaskB/subtaskB_dev.jsonl --prediction_file_path ./predictions/subtaskB_roberta.jsonl --subtask B --model roberta-base

#python3 subtaskB/baseline/transformer_baseline.py --train_file_path ./data/SubtaskB/subtaskB_train.jsonl --test_file_path ./data/SubtaskB/subtaskB_dev.jsonl --prediction_file_path ./predictions/subtaskB_xlmr.jsonl --subtask B --model xlm-roberta-base

#python3 subtaskB/baseline/transformer_baseline.py --train_file_path ./data/SubtaskB/subtaskB_train.jsonl --test_file_path ./data/SubtaskB/subtaskB_dev.jsonl --prediction_file_path ./predictions/subtaskB_mistral.jsonl --subtask B --model mistralai/Mistral-7B-v0.1


# python llama_seq_clf.py subtaskA_mono 7b 
python llama_seq_clf.py subtaskA_mul 7b 
python llama_seq_clf.py subtaskB 7b 

