# -*- coding: utf-8 -*-

import sys
import json
from typing import List, Any, Dict
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers.data import  *
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

import torch  
torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)

import evaluate

from modeling_llama import LlamaForSequenceClassification

def load_subtaskA_mono():
    ret = {}
    for split_name in ['train', 'dev', 'test']:
        data = []
        with open(f'./data/SubtaskA/subtaskA_{split_name}_monolingual.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

def load_subtaskA_mul():
    ret = {}
    for split_name in ['train', 'dev', 'test']:
        data = []
        with open(f'./data/SubtaskA/subtaskA_{split_name}_multilingual.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

def load_subtaskB():
    ret = {}
    for split_name in ['train', 'dev', 'test']:
        data = []
        with open(f'./data/SubtaskB/subtaskB_{split_name}.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

if len(sys.argv) != 3:
    print('usage python %.py dataset model_size')
    sys.exit()


dataset, model_size = sys.argv[1], sys.argv[2]
epochs = 10
batch_size = 16
learning_rate = 5e-5
lora_r = 12
max_length = 128
if model_size.lower() == '7b':
    model_id = 'NousResearch/Llama-2-7b-hf'
elif model_size.lower() == '13b':
    model_id = 'NousResearch/Llama-2-13b-hf'

test_name = 'test'
text_name = None

##############################################################################################################################################################################
if dataset == 'subtaskA_mono':
    id2label = {0: "human", 1: "machine"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_subtaskA_mono()
    dev_name = 'dev'
    text_name = 'text'
elif dataset == 'subtaskA_mul':
    id2label = {0: "human-written", 1: "machine-generated"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_subtaskA_mul()
    dev_name = 'dev'
    text_name = 'text'
elif dataset == 'subtaskB':
    id2label = {0: "human", 1: "chatGPT", 2: "cohere", 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_subtaskB()
    dev_name = 'dev'
    text_name = 'text'
else:
    raise NotImplementedError
##############################################################################################################################################################################
accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForSequenceClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
).bfloat16()
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    global text_name
    if isinstance(text_name, str):
        d = examples[text_name]
    else:
        d = examples[text_name[0]]
        for n in text_name[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t

    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)


tokenized_ds = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="clf",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds[dev_name],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Predictions on the test set
predictions = trainer.predict(tokenized_ds[test_name])

# Save the predictions to a file
with open(f"{dataset}_{model_size}_test_predictions.json", "w") as f:
    json.dump(predictions.predictions.tolist(), f)
