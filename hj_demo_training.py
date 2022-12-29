from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import Trainer


def tokenize_function(examples):
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset():
    dataset = load_dataset("yelp_review_full")
    print('dataset[train][100]: ', dataset["train"][100])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    return small_train_dataset, small_eval_dataset


def main():
    small_train_dataset, small_eval_dataset = prepare_dataset()

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("finished...")


if __name__ == '__main__':
    main()