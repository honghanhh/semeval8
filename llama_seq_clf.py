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
    for split_name in ['train', 'dev']:
        data = []
        with open(f'./data/SubtaskA/subtaskA_{split_name}_monolingual.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

def load_subtaskA_mul():
    ret = {}
    for split_name in ['train', 'dev']:
        data = []
        with open(f'./data/SubtaskA/subtaskA_{split_name}_multilingual.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

def load_subtaskB():
    ret = {}
    for split_name in ['train', 'dev']:
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
if dataset == 'agnews':
    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_dataset("ag_news")
    text_name = 'text'
elif dataset == 'twitterfin':
    id2label = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    test_name = 'validation'
    text_name = 'text'
elif dataset == 'sst2':
    id2label = {0: "negative", 1: "positive"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_dataset("sst2")
    test_name = 'validation'
    text_name = 'sentence'
##############################################################################################################################################################################
elif dataset == 'subtaskA_mono':
    id2label = {0: "human", 1: "machine"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_subtaskA_mono()
    test_name = 'dev'
    text_name = 'text'
elif dataset == 'subtaskA_mul':
    id2label = {0: "human-written", 1: "machine-generated"}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_subtaskA_mul()
    test_name = 'dev'
    text_name = 'text'
elif dataset == 'subtaskB':
    id2label = {0: "human", 1: "chatGPT", 2: "cohere", 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_subtaskB()
    test_name = 'dev'
    text_name = 'text'
##############################################################################################################################################################################
elif dataset in ['amazon_de', 'amazon_en', 'amazon_es', 'amazon_fr', 'amazon_ja', 'amazon_zh']:
    max_length = 200
    batch_size = 4
    lang = dataset.split('_')[1]
    id2label = {0: 'furniture', 1: 'baby_product', 2: 'jewelry', 3: 'musical_instruments', 4: 'industrial_supplies', 5: 'pc', 6: 'other', 7: 'pet_products', 8: 'book', 9: 'apparel', 10: 'automotive', 11: 'digital_video_download', 12: 'beauty', 13: 'toy', 14: 'shoes', 15: 'personal_care_appliances', 16: 'camera', 17: 'digital_ebook_purchase', 18: 'watch', 19: 'drugstore', 20: 'grocery', 21: 'kitchen', 22: 'home', 23: 'office_product', 24: 'home_improvement', 25: 'electronics', 26: 'video_games', 27: 'sports', 28: 'luggage', 29: 'lawn_and_garden', 30: 'wireless'}
    label2id = {v: k for k, v in id2label.items()}
    ds = load_dataset("amazon_reviews_multi", lang)
    ds = ds.rename_column('product_category', 'label')
    text_name = ['review_title', 'review_body']
    # reimplement DataCollatorWithPaddingAmazon
    class DataCollatorWithPaddingAmazon(DataCollatorWithPadding):
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # print('>>> features>>>', features)
            new_features = []
            for v in features:
                label = v.pop('label')
                v['label'] = label2id[label]
                new_features.append(v)
            features = new_features
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]
            return batch

    DataCollatorWithPadding = DataCollatorWithPaddingAmazon
else:
    raise NotImplementedError

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
    save_strategy="no",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds[test_name],
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
