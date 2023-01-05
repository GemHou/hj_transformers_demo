from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader


def tokenize_function_bert(examples):
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def tokenize_function_xlnet(examples):
    model_name = "xlnet-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset():
    # yelp
    dataset_yelp = load_dataset("yelp_review_full")
    dataset_yelp["train"] = dataset_yelp["train"].select(range(1000))
    dataset_yelp["test"] = dataset_yelp["test"].select(range(1000))
    dataset_tokenized_yelp = dataset_yelp.map(tokenize_function_bert, batched=True)
    dataset_tokenized_yelp = dataset_tokenized_yelp.remove_columns(["text"])
    dataset_tokenized_yelp = dataset_tokenized_yelp.rename_column("label", "labels")  # important!!!!!!!!!!!!!!!!!!!!!!
    dataset_tokenized_yelp.set_format("torch")
    dataset_tokenized_yelp_train = dataset_tokenized_yelp["train"].shuffle(seed=42)  # .select(range(1000))
    dataset_tokenized_yelp_test = dataset_tokenized_yelp["test"].shuffle(seed=42)  # .select(range(1000))
    dataloader_yelp_train = DataLoader(dataset_tokenized_yelp_train, shuffle=True, batch_size=6)
    dataloader_yelp_eval = DataLoader(dataset_tokenized_yelp_test, shuffle=True, batch_size=10)

    # wikipedia
    dataset_wikipedia = load_dataset("wikipedia", '20220301.simple', beam_runner='DirectRunner')
    dataset_wikipedia["train"] = dataset_wikipedia["train"].select(range(1000))
    dataset_tokenized_wikipedia = dataset_wikipedia.map(tokenize_function_xlnet, batched=True)
    dataset_tokenized_wikipedia = dataset_tokenized_wikipedia.remove_columns(["id"])
    dataset_tokenized_wikipedia = dataset_tokenized_wikipedia.remove_columns(["url"])
    dataset_tokenized_wikipedia = dataset_tokenized_wikipedia.remove_columns(["title"])
    dataset_tokenized_wikipedia = dataset_tokenized_wikipedia.remove_columns(["text"])
    dataset_tokenized_wikipedia.set_format("torch")
    dataset_tokenized_wikipedia = dataset_tokenized_wikipedia["train"].shuffle(seed=42)  # .select(range(1000))
    dataloader_wikipedia = DataLoader(dataset_tokenized_wikipedia, shuffle=True, batch_size=1)

    # bookcorpus
    # dataset_bookcorpus = load_dataset("bookcorpus")

    return [dataloader_yelp_train, dataloader_yelp_eval], \
           [dataloader_wikipedia]


DATA_NUM = 1000  # None 1000
BLOCK_SIZE = 128


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


def main():
    [dataloader_yelp_train, dataloader_yelp_eval], \
        [dataloader_wikipedia] = prepare_dataset()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_yelp_train = next(iter(dataloader_yelp_train))
    batch_yelp_train = {k: v.to(device) for k, v in batch_yelp_train.items()}
    batch_wikipedia = next(iter(dataloader_wikipedia))
    batch_wikipedia = {k: v.to(device) for k, v in batch_wikipedia.items()}

    print("finish")


if __name__ == '__main__':
    main()
