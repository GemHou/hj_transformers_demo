from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import os
import re
import time
import random


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


DATA_NUM = 10  # None 1000
BLOCK_SIZE = 128


def tokenize_function_wikipedia(examples):
    model_name = "distilgpt2"  # bert-base-cased distilgpt2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    temp_1 = [" ".join(x) for x in examples["text"]]
    temp2 = tokenizer(temp_1, truncation=True)
    return temp2  # , padding="max_length"


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
    batch = wikipedia["train"][0]
    print("batch: ", batch)
    tokenized_wikipedia = wikipedia.map(tokenize_function_wikipedia,
                                        batched=True,
                                        num_proc=8,
                                        remove_columns=wikipedia["train"].column_names)  # time!!!!!!!!!!!!!!!!!!!!!!!!
    batch = tokenized_wikipedia["train"][0]
    print("batch: ", batch)
    lm_wikipedia = tokenized_wikipedia.map(group_texts, batched=True, num_proc=8)
    batch = tokenized_wikipedia["train"][0]
    print("batch: ", batch)
    lm_wikipedia.set_format("torch")
    train_dataloader_wikipedia = DataLoader(lm_wikipedia["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader_wikipedia = DataLoader(lm_wikipedia["test"], shuffle=True, batch_size=batch_size)
    lm_dataset = lm_wikipedia
    train_dataloader = train_dataloader_wikipedia
    eval_dataloader = eval_dataloader_wikipedia
    return eval_dataloader, lm_dataset, train_dataloader


def list_files_with_extensions(temp_dir, extensions):
    return [f for f in os.listdir(temp_dir) if f.endswith(extensions)]


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
                l.replace('\n', '')
                for prot in data.split('>') for l in prot.strip().split('\n', 1)
            ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def hj_group_text(data_tokenizer):
    start_time = time.time()
    concatenated_examples = {k: sum(data_tokenizer[k], []) for k in data_tokenizer.keys()}
    total_length = len(concatenated_examples[list(data_tokenizer.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    results = {
        k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)] for k, t in
        concatenated_examples.items()
    }
    print("group text time: ", time.time()-start_time)
    return results


def process_repeated_data(seq_datasets):
    data_num = len(seq_datasets)
    print("original data_num: ", data_num)
    seq_datasets = list(set(seq_datasets))
    random.shuffle(seq_datasets)
    data_num = len(seq_datasets)
    print("precessed data_num: ", data_num)
    return seq_datasets


class HjDataset(torch.utils.data.Dataset):
    def process_batch_data(self, data_list):
        start_time = time.time()

        data_space = [" ".join(x) for x in data_list]
        model_name = "distilgpt2"  # bert-base-cased distilgpt2
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_tokenizer = tokenizer(data_space, truncation=True)

        # results1 = hj_group_text(data_tokenizer)
        results2 = data_tokenizer.data
        new_input_ids = list()
        new_attention_mask = list()
        for i in range(len(results2['input_ids'])):
            i_input = results2['input_ids'][i]
            i_attention = results2['attention_mask'][i]
            num = len(i_input) / 128
            # print("num: ", num)
            if num >= 1:
                for j in range(int(num)):
                    new_input = i_input[128*j:128*(j+1)]
                    assert len(new_input) == 128
                    new_input_ids.append(new_input)
                    new_attention_mask.append(i_attention[128*j:128*(j+1)])
                new_input = i_input[-129:-1]
                if len(new_input) == 128:
                    new_input_ids.append(new_input)
                    new_attention_mask.append(i_attention[-129:-1])
        results2['input_ids'] = new_input_ids
        results2['attention_mask'] = new_attention_mask

        data_data = results2

        # data_data = data_tokenizer.data
        if self.output_format == "list":
            pass
        elif self.output_format == "torch":
            data_data['input_ids'] = torch.tensor(data_data['input_ids'])
            data_data['attention_mask'] = torch.tensor(data_data['attention_mask'])
        else:
            raise
        data_data['labels'] = data_data['input_ids']
        data_final = data_data

        print("process_batch_data time: ", time.time() - start_time)
        return data_final

    def __init__(self, train_test_mode="train"):
        # self.hj_data = ["abcde", "12345", "xyzxy", "56789", "98765"]
        if train_test_mode == "train":
            """
            self.hj_data = ["GHSKMSDVKCTSVVLLSVLQQLRVESSSKLWAQCVQLHNDILLAKDTTEAFEKMVSLLSVLLSMQGAVDINRLCEEMLDNRATLQ",
                            "VILPNNDRHQITDTTNGHYAPVTYIQVEAPTGTFIASGVVVGKDTLLTNKHVVDATHGDPHALKAFPSAINQDNYPNGGFTAEQITKYSGEGDLAIVKFSPNEQNKHIGEVVKPATMSNNAETQTNQNITVTGYPGDKPVATMWESKGKITYLKGEAMQYDLSTTGGNSGSPVFNEKNEVIGIHWGGVPNEFNGAVFINENVRNFLKQNIEDINFANDDQPNNPDNPDNPNNPDNPNNPDNPNNPDEPNNPDNPNNPDNPDNGDNNNSDNPDAA",
                            "MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFEILERR",
                            "MASMAKKDVIELEGTVSEALPNAMFKVKLENGHEILCHISGKLRMNFIRILEGDKVNVELSPYDLTRGRITWRKKLEHHHHHH",
                            "QYDDHPPVFQKKFYIGGVSEDARMFASVLRVKATDRDTGNYSAMAYRLIIPPIKEGKEGFVVETYTGLIKTAMLFHNMRRSYFKFQVIATDDYGKGLSGKADVLVSVVNQLDMQVIVSNVPPTLVEKKIEDLTEILDRYVQEQIPGAKVVVESIGARRHGDAYSLEDYSKCDLTVYAIDPQTNRAIDRNELFKFLDGKLLDINKDFQPYYGEGGRILEIRTPEAVTSIKKRGESLGYTEGASRLVPR",
                            "GPGSDSPDRPWNPPTFSPALLVVTEGDNATFTCSFSNTSESFHVVWHRESPSGQTDTLAAFPEDRSQPGQDARFRVTQLPNGRDFHMSVVRARRNDSGTYVCGVISLAPKIQIKESLRAELRVTERAAA",
                            "XADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK",
                            "MKGDTKVINYLNKLLGNELVAINQYFLHARMFKNWGLKRLNDVEYHESIDEMKHADRYIERILFLEGLPNLQDLGKLNIGEDVEEMLRSDLALELDGAKNLREAIGYADSVHDYVSRDMMIEILRDEEGHIDWLETELDLIQKMGLQNYLQAQIREEG",
                            "SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN",
                            ]
            """
            fasta_dir = "/home/gemhou/Study/data"
            seqs = None
            for fasta_file in list_files_with_extensions(fasta_dir, (".fasta", ".fa")):
                with open(os.path.join(fasta_dir, fasta_file), "r") as fp:
                    data = fp.read()
                tags, seqs = parse_fasta(data)
            seq_datasets = seqs
            seq_datasets = seq_datasets[:int(len(seq_datasets)*0.8)]
            seq_datasets = process_repeated_data(seq_datasets)
            self.hj_data = seq_datasets # 1000:0.6s 10000:143s

        elif train_test_mode == "test":
            """
            self.hj_data = ["MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
                            "MNTPEHMTAVVQRYVAALNAGDLDGIVALFADDATVENPVGSEPRSGTAAIREFYANSLKLPLAVELTQEVRAVANEAAFAFIVSFEYQGRKTVVAPIDHFRFNGAGKVVSMRALFGEKNIHAGA",
                            ]
            """

            fasta_dir = "/home/gemhou/Study/data"
            seqs = None
            for fasta_file in list_files_with_extensions(fasta_dir, (".fasta", ".fa")):
                with open(os.path.join(fasta_dir, fasta_file), "r") as fp:
                    data = fp.read()
                tags, seqs = parse_fasta(data)
            seq_datasets = seqs
            seq_datasets = seq_datasets[int(len(seq_datasets)*0.8):]
            seq_datasets = process_repeated_data(seq_datasets)
            self.hj_data = seq_datasets
        else:
            self.hj_data = None
            raise

        self.output_format = "torch"
        print("process_batch_data waiting......")
        self.processed_data = self.process_batch_data(self.hj_data)
        print("debug")

    def __len__(self):
        return len(self.processed_data["input_ids"])

    def process_data(self, data_raw):
        data_list = [data_raw]
        data_space = [" ".join(x) for x in data_list]
        model_name = "distilgpt2"  # bert-base-cased distilgpt2
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_tokenizer = tokenizer(data_space, truncation=True)

        results = hj_group_text(data_tokenizer)
        data_data = results

        # data_data = data_tokenizer.data
        data_data['input_ids'] = data_data['input_ids'][0]
        data_data['attention_mask'] = data_data['attention_mask'][0]
        if self.output_format == "list":
            pass
        elif self.output_format == "torch":
            data_data['input_ids'] = torch.tensor(data_data['input_ids'])
            data_data['attention_mask'] = torch.tensor(data_data['attention_mask'])
        else:
            raise
        data_data['labels'] = data_data['input_ids']
        data_final = data_data
        return data_final

    def __getitem__(self, item):
        # data_raw = self.hj_data[item]
        # data_final_1 = self.process_data(data_raw)

        data_final_2 = dict()
        # print("item: ", item)
        data_final_2["input_ids"] = self.processed_data["input_ids"][item]
        data_final_2["attention_mask"] = self.processed_data["attention_mask"][item]
        data_final_2["labels"] = self.processed_data["labels"][item]

        return data_final_2

    def set_format(self, output_format):
        if output_format == "torch":
            self.output_format = output_format


def main():
    """
    [dataloader_yelp_train, dataloader_yelp_eval], \
        [dataloader_wikipedia] = prepare_dataset()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_yelp_train = next(iter(dataloader_yelp_train))
    batch_yelp_train = {k: v.to(device) for k, v in batch_yelp_train.items()}
    batch_wikipedia = next(iter(dataloader_wikipedia))
    batch_wikipedia = {k: v.to(device) for k, v in batch_wikipedia.items()}

    print("finish")
    """
    # eval_dataloader, lm_dataset, train_dataloader = prepare_wikipedia_dataset(batch_size=2)
    # lm_train = lm_dataset["train"]
    # batch_wikipedia = next(iter(eval_dataloader))

    hj_dataset = HjDataset()
    hj_dataset.set_format("torch")
    train_dataloader_hj = DataLoader(hj_dataset, shuffle=True, batch_size=2)

    # batch.pop('attention_mask')
    batch_hj = next(iter(train_dataloader_hj))
    print("batch_hj: ", batch_hj)
    print("finished...................")


if __name__ == '__main__':
    main()
