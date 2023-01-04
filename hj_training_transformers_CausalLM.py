import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
# from transformers import DataCollatorWithPadding
from transformers import DataCollatorForLanguageModeling

import evaluate

block_size = 128


def tokenize_function(examples):
    model_name = "distilgpt2"  # bert-base-cased distilgpt2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)  # , padding="max_length"


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    results = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    results["labels"] = results["input_ids"].copy()
    return results


def prepare_dataset():
    eli5 = load_dataset("eli5", split="train_asks[:1000]")
    eli5 = eli5.train_test_split(test_size=0.2)
    print("eli5[train][0]: ", eli5["train"][0])
    eli5 = eli5.flatten()
    print("eli5[train][0]: ", eli5["train"][0])

    tokenized_eli5 = eli5.map(tokenize_function,
                              batched=True,
                              num_proc=4,
                              remove_columns=eli5["train"].column_names)

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    return lm_dataset


def main():
    lm_dataset = prepare_dataset()

    print("lm_dataset[train][0]: ", lm_dataset["train"][0])
    # batch = lm_dataset["train"][0]

    model_name = "distilgpt2"  # bert-base-cased distilgpt2 distilroberta-base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # bert-base-cased

    training_args = TrainingArguments(output_dir="my_awesome_eli5_clm-model",
                                      evaluation_strategy="epoch",
                                      learning_rate=2e-5,
                                      weight_decay=0.01,
                                      # per_device_train_batch_size=1
                                      # auto_find_batch_size=1
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        # compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    print("finished...")


if __name__ == '__main__':
    main()