from datasets import load_dataset
import numpy as np
import evaluate
import time

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def tokenize_function(examples):
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset():
    # yelp
    dataset_yelp = load_dataset("yelp_review_full")
    dataset_yelp["train"] = dataset_yelp["train"].select(range(1000))
    dataset_yelp["test"] = dataset_yelp["test"].select(range(1000))
    dataset_yelp = dataset_yelp.map(tokenize_function, batched=True)
    dataset_yelp = dataset_yelp.remove_columns(["text"])
    dataset_yelp = dataset_yelp.rename_column("label", "labels")
    dataset_yelp.set_format("torch")
    dataset_yelp_train = dataset_yelp["train"].shuffle(seed=42)  # .select(range(1000))
    dataset_yelp_test = dataset_yelp["test"].shuffle(seed=42)  # .select(range(1000))
    dataloader_yelp_train = DataLoader(dataset_yelp_train, shuffle=True, batch_size=6)
    dataloader_yelp_eval = DataLoader(dataset_yelp_test, shuffle=True, batch_size=10)

    # wikipedia
    dataset_wikipedia = load_dataset("wikipedia", '20220301.simple', beam_runner='DirectRunner')
    dataset_wikipedia["train"] = dataset_wikipedia["train"].select(range(1000))
    dataset_wikipedia = dataset_wikipedia.map(tokenize_function, batched=True)
    dataset_wikipedia.set_format("torch")
    dataset_wikipedia = dataset_wikipedia["train"].shuffle(seed=42)  # .select(range(1000))
    dataloader_wikipedia  = DataLoader(dataset_wikipedia, shuffle=True, batch_size=6)

    # bookcorpus
    # dataset_bookcorpus = load_dataset("bookcorpus")

    return [dataset_yelp_train, dataset_yelp_test, dataloader_yelp_train, dataloader_yelp_eval], \
           [dataset_wikipedia, dataloader_wikipedia]


def main():
    [dataset_yelp_train, dataset_yelp_test, dataloader_yelp_train, dataloader_yelp_eval], \
        [dataset_wikipedia, dataloader_wikipedia] = prepare_dataset()

    batch_yelp_train = next(iter(dataloader_yelp_train))
    batch_wikipedia = next(iter(dataloader_wikipedia))

    print("finish")


if __name__ == '__main__':
    main()
