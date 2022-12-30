from datasets import load_dataset
import numpy as np
import evaluate
import matplotlib.pyplot as plt

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
    dataset = load_dataset("yelp_review_full")
    # print('dataset[train][100]: ', dataset["train"][100])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=7)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

    return small_train_dataset, small_eval_dataset, train_dataloader, eval_dataloader


def train_iter(device, lr_scheduler, model, num_epochs, optimizer,
               progress_bar, train_dataloader, train_loss_list, writer):
    global_step = 0
    for epoch in range(num_epochs):
        """"""
        model.train()
        for batch in train_dataloader:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # print("loss: ", loss)
            train_loss = loss.item()
            train_loss_list.append(train_loss)
            # plt.subplot(2, 1, 1)
            # plt.cla()
            # plt.plot(train_loss_list)
            # plt.pause(0.00000000000000001)

            writer.add_scalar('train_loss', train_loss, global_step)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        """
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            plt.subplot(2, 1, 2)
            plt.cla()
            test_loss = predictions.item()
            test_loss_list.append(test_loss)
            plt.plot(test_loss_list)
            plt.pause(0.00000000000000001)

            metric.add_batch(predictions=predictions, references=batch["labels"])
        metric.compute()
        """


def main():
    small_train_dataset, small_eval_dataset, train_dataloader, eval_dataloader = prepare_dataset()

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)
    model.to(device)
    train_loss_list = []
    test_loss_list = []
    progress_bar = tqdm(range(num_training_steps))
    metric = evaluate.load("accuracy")

    writer = SummaryWriter('./tensorboard')

    train_iter(device, lr_scheduler, model, num_epochs, optimizer,
               progress_bar, train_dataloader, train_loss_list, writer)

    plt.show()

    print("finished...")


if __name__ == '__main__':
    main()