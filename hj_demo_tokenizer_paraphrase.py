from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def generate_results(model, sequence1, sequence2, tokenizer):
    paraphrase = tokenizer(sequence1, sequence2, return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase).logits
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    return paraphrase_results


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

    classes = ["not paraphrase", "is paraphrase"]

    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    paraphrase_results_02 = generate_results(model, sequence_0, sequence_2, tokenizer)
    paraphrase_results_01 = generate_results(model, sequence_0, sequence_1, tokenizer)

    for i in range(len(classes)):
        print(f"{classes[i]}: {int(round(paraphrase_results_02[i] * 100))}%")

    for i in range(len(classes)):
        print(f"{classes[i]}: {int(round(paraphrase_results_01[i] * 100))}%")


if __name__ == '__main__':
    main()
