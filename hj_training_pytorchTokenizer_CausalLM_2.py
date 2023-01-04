from datasets import load_dataset
import numpy as np
import evaluate
import time

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

block_size = 128
BATCH_SIZE_LIST = [4096]


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


def prepare_dataset(batch_size):
    eli5 = load_dataset("eli5", split="train_asks[:5000]")  # 50000 1000
    eli5 = eli5.train_test_split(test_size=0.2)
    print("eli5[train][0]: ", eli5["train"][0])
    eli5 = eli5.flatten()
    print("eli5[train][0]: ", eli5["train"][0])

    tokenized_eli5 = eli5.map(tokenize_function,
                              batched=True,
                              num_proc=4,
                              remove_columns=eli5["train"].column_names)

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    lm_dataset.set_format("torch")

    train_dataloader = DataLoader(lm_dataset["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(lm_dataset["test"], shuffle=True, batch_size=batch_size)

    batch = next(iter(train_dataloader))

    return lm_dataset, train_dataloader, eval_dataloader


def train_iter(device, lr_scheduler, model, num_epochs, optimizer,
               progress_bar, train_dataloader, eval_dataloader, writer, metric, batch_size):
    train_global_step = 0
    model.train()
    for epoch in range(num_epochs):
        """"""
        for batch in train_dataloader:
            start_time = time.time()

            train_global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # print("loss: ", loss)
            train_loss = loss.item()

            writer.add_scalar('loss/train_loss', train_loss, train_global_step)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            batch_time = time.time() - start_time

            """"""

            if train_global_step % 2 == 0:

                model.eval()
                # for batch in eval_dataloader:
                batch = next(iter(eval_dataloader))
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                test_loss = outputs.loss.item()
                writer.add_scalar('loss/test_loss', test_loss, train_global_step)

                """
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
                metric_value = metric.compute()
                accuracy = metric_value['accuracy']
                # print("accuracy: ", accuracy)
                writer.add_scalar('accuracy', accuracy, train_global_step)
                """

                model.train()

                writer.add_scalar('others/batch_size', batch_size, train_global_step)
                writer.add_scalar('others/batch_time', batch_time, train_global_step)

            """"""


def get_time_str():
    local_time = time.localtime(time.time())
    # date_str = str(local_time[0]) + str(local_time[1]) + str(local_time[2])
    date_str1 = str(local_time[0])
    if len(date_str1) == 1:
        date_str1 = "0" + date_str1
    date_str2 = str(local_time[1])
    if len(date_str2) == 1:
        date_str2 = "0" + date_str2
    date_str3 = str(local_time[2])
    if len(date_str3) == 1:
        date_str3 = "0" + date_str3
    date_str = date_str1 + date_str2 + date_str3

    time_str1 = str(local_time[3])
    if len(time_str1) == 1:
        time_str1 = "0" + time_str1
    time_str2 = str(local_time[4])
    if len(time_str2) == 1:
        time_str2 = "0" + time_str2
    time_str3 = str(local_time[5])
    if len(time_str3) == 1:
        time_str3 = "0" + time_str3
    time_str = time_str1 + time_str2 + time_str3
    return date_str, time_str


def main():
    for batch_size in BATCH_SIZE_LIST:
        lm_dataset, train_dataloader, eval_dataloader = prepare_dataset(batch_size)

        model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # bert-base-cased

        optimizer = AdamW(model.parameters(), lr=2e-5)

        num_epochs = 1
        num_training_steps = num_epochs * len(train_dataloader)  # lm_dataset["train"]
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                     num_training_steps=num_training_steps)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("device: ", device)
        model.to(device)
        progress_bar = tqdm(range(num_training_steps))
        metric = evaluate.load("accuracy")

        date_str, time_str = get_time_str()
        writer = SummaryWriter('./tensorboard/' + date_str + "_" + time_str)

        train_iter(device, lr_scheduler, model, num_epochs, optimizer,
                   progress_bar, train_dataloader, eval_dataloader, writer, metric, batch_size)

        print("finished...")


if __name__ == '__main__':
    main()
