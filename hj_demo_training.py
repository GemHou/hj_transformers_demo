from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_dataset_1():
    dataset = load_dataset("yelp_review_full")
    print('dataset[train][100]: ', dataset["train"][100])
    return dataset


def prepare_dataset_2(dataset):
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    return small_train_dataset, small_eval_dataset


def main():
    dataset = prepare_dataset_1()

    small_train_dataset, small_eval_dataset = prepare_dataset_2(dataset)

    print("finished...")


if __name__ == '__main__':
    main()