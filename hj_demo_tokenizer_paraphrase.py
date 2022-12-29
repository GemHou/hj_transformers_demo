from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def main():
    print("hello world...")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")


if __name__ == '__main__':
    main()
