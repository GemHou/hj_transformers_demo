from datasets import load_dataset, DatasetDict
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
from hj_inference_textGeneration_ptFile import test_success_rate

BLOCK_SIZE = 128
BATCH_SIZE_LIST = [4]
DATASET_NAME = "hj"  # eli5 wikipedia hj
DATA_NUM = None  # None 1000
LOAD_FLAG = False  # True False

if DATA_NUM is not None:
    assert DATA_NUM >= min(BATCH_SIZE_LIST)


def tokenize_function_eli5(examples):
    model_name = "distilgpt2"  # bert-base-cased distilgpt2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)  # , padding="max_length"


def tokenize_function_wikipedia(examples):
    model_name = "distilgpt2"  # bert-base-cased distilgpt2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenizer([" ".join(x) for x in examples["text"]], truncation=True)  # , padding="max_length"


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    results = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)] for k, t in concatenated_examples.items()
    }
    results["labels"] = results["input_ids"].copy()
    return results


def prepare_eli5_dataset(batch_size):
    # eli5_origin = load_dataset("eli5")
    if DATA_NUM is None:
        eli5_train = load_dataset("eli5", split="train_asks[:]").shuffle(seed=42)  # 50000 1000
        eli5_valid = load_dataset("eli5", split="validation_asks[:]").shuffle(seed=42)
    else:
        eli5_train = load_dataset("eli5", split="train_asks[:" + str(DATA_NUM) + "]").shuffle(seed=42)  # 50000 1000
        eli5_valid = load_dataset("eli5", split="validation_asks[:" + str(DATA_NUM) + "]").shuffle(seed=42)
    eli5 = DatasetDict()
    eli5["train"] = eli5_train
    eli5["test"] = eli5_valid
    # eli5 = eli5.train_test_split(test_size=0.2)
    print("eli5[train][0]: ", eli5["train"][0])
    eli5 = eli5.flatten()
    print("eli5[train][0]: ", eli5["train"][0])
    tokenized_eli5 = eli5.map(tokenize_function_eli5,
                              batched=True,
                              num_proc=8,
                              remove_columns=eli5["train"].column_names)  # time!!!!!!!!!!!!!!!!!!!!!!!!
    print("tokenized_eli5[train][0]: ", tokenized_eli5["train"][0])
    lm_eli5 = tokenized_eli5.map(group_texts, batched=True, num_proc=8)
    lm_eli5.set_format("torch")
    train_dataloader_eli5 = DataLoader(lm_eli5["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader_eli5 = DataLoader(lm_eli5["test"], shuffle=True, batch_size=batch_size)
    batch = next(iter(train_dataloader_eli5))
    lm_dataset = lm_eli5
    train_dataloader = train_dataloader_eli5
    eval_dataloader = eval_dataloader_eli5
    return eval_dataloader, lm_dataset, train_dataloader


def prepare_wikipedia_dataset(batch_size):
    if DATA_NUM is None:
        wikipedia = load_dataset("wikipedia", '20220301.simple', beam_runner='DirectRunner',
                                 split="train[:]").shuffle(seed=42)
    else:
        wikipedia = load_dataset("wikipedia", '20220301.simple', beam_runner='DirectRunner',
                                 split="train[:" + str(DATA_NUM) + "]").shuffle(seed=42)
    wikipedia = wikipedia.train_test_split(test_size=0.2)
    print("wikipedia[train][0]: ", wikipedia["train"][0])
    tokenized_wikipedia = wikipedia.map(tokenize_function_wikipedia,
                                        batched=True,
                                        num_proc=8,
                                        remove_columns=wikipedia["train"].column_names)  # time!!!!!!!!!!!!!!!!!!!!!!!!
    print("tokenized_wikipedia[train][0]: ", tokenized_wikipedia["train"][0])
    lm_wikipedia = tokenized_wikipedia.map(group_texts, batched=True, num_proc=8)
    print("tokenized_wikipedia[train][0]: ", tokenized_wikipedia["train"][0])
    lm_wikipedia.set_format("torch")
    train_dataloader_wikipedia = DataLoader(lm_wikipedia["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader_wikipedia = DataLoader(lm_wikipedia["test"], shuffle=True, batch_size=batch_size)
    lm_dataset = lm_wikipedia
    train_dataloader = train_dataloader_wikipedia
    eval_dataloader = eval_dataloader_wikipedia
    return eval_dataloader, lm_dataset, train_dataloader


def prepare_hj_dataset(batch_size):
    from hj_dataset import HjDataset
    hj_dataset = HjDataset()
    hj_dataset_test = HjDataset(train_test_mode="test")
    hj_dataset.set_format("torch")
    train_dataloader = DataLoader(hj_dataset, shuffle=True, batch_size=batch_size)

    eval_dataloader = DataLoader(hj_dataset_test, shuffle=True, batch_size=batch_size)

    # train_dataloader = train_dataloader_hj
    # eval_dataloader = eval_dataloader_hj
    lm_dataset = hj_dataset

    """
    if DATA_NUM is None:
        wikipedia = load_dataset("wikipedia", '20220301.simple', beam_runner='DirectRunner',
                                 split="train[:]").shuffle(seed=42)
    else:
        wikipedia = load_dataset("wikipedia", '20220301.simple', beam_runner='DirectRunner',
                                 split="train[:" + str(DATA_NUM) + "]").shuffle(seed=42)
    wikipedia = wikipedia.train_test_split(test_size=0.2)
    print("wikipedia[train][0]: ", wikipedia["train"][0])
    tokenized_wikipedia = wikipedia.map(tokenize_function_wikipedia,
                                        batched=True,
                                        num_proc=8,
                                        remove_columns=wikipedia["train"].column_names)  # time!!!!!!!!!!!!!!!!!!!!!!!!
    print("tokenized_wikipedia[train][0]: ", tokenized_wikipedia["train"][0])
    lm_wikipedia = tokenized_wikipedia.map(group_texts, batched=True, num_proc=8)
    print("tokenized_wikipedia[train][0]: ", tokenized_wikipedia["train"][0])
    lm_wikipedia.set_format("torch")
    train_dataloader_wikipedia = DataLoader(lm_wikipedia["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader_wikipedia = DataLoader(lm_wikipedia["test"], shuffle=True, batch_size=batch_size)
    lm_dataset = lm_wikipedia
    train_dataloader = train_dataloader_wikipedia
    eval_dataloader = eval_dataloader_wikipedia
    """
    return eval_dataloader, lm_dataset, train_dataloader


def prepare_dataset(batch_size):
    if DATASET_NAME == "eli5":
        eval_dataloader, lm_dataset, train_dataloader = prepare_eli5_dataset(batch_size)
    elif DATASET_NAME == "wikipedia":
        eval_dataloader, lm_dataset, train_dataloader = prepare_wikipedia_dataset(batch_size)
    elif DATASET_NAME == "hj":
        eval_dataloader, lm_dataset, train_dataloader = prepare_hj_dataset(batch_size)
    else:
        lm_dataset, train_dataloader, eval_dataloader = None, None, None
        raise

    return lm_dataset, train_dataloader, eval_dataloader


def train_iter(device, lr_scheduler, model, num_epochs, optimizer,
               progress_bar, train_dataloader, eval_dataloader, writer, metric, batch_size):
    train_global_step = 0
    model.train()
    last_time = time.time()
    model_name = "distilgpt2"  # bert-base-cased-finetuned-mrpc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for epoch in range(num_epochs):
        # torch.nn.init.xavier_normal(model.transformer.weight)
        """"""
        for batch in train_dataloader:
            # batch.pop('attention_mask')  # attention_mask labels input_ids

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

            now_time = time.time()
            batch_time = now_time - last_time
            last_time = now_time

            """"""

            if train_global_step % 100 == 1:

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

                temp = tokenizer.decode(batch["input_ids"][0])
                temp = temp.replace(" ", "")
                init_obs_acid = temp[:int(len(temp)*0.3)]
                true_acid = temp[int(len(temp)*0.3):]
                init_obs_acid = "".join(init_obs_acid)
                true_acid = "".join(true_acid)

                if train_global_step % 1000 == 1:
                    model.to("cpu")
                    success_rate = test_success_rate(init_obs_acid, model, tokenizer, true_acid)
                    model.to("cuda:0")

                    writer.add_scalar('eval/success_rate', success_rate, train_global_step)

                writer.add_scalar('others/batch_size', batch_size, train_global_step)
                writer.add_scalar('others/batch_time', batch_time, train_global_step)

                model.train()

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
        if LOAD_FLAG:
            model.load_state_dict(torch.load("../para_temp.pt"))

        optimizer = AdamW(model.parameters(), lr=2e-5)

        num_epochs = 20
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
        torch.save(model.state_dict(), "../para_temp.pt")

        print("finished...")


if __name__ == '__main__':
    main()
