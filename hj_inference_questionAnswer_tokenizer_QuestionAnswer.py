from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


def generate_results(model, sequence1, sequence2, tokenizer):
    paraphrase = tokenizer(sequence1, sequence2, return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase).logits
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    return paraphrase_results


def prepare_task():
    text = r"""

    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose

    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural

    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between

    TensorFlow 2.0 and PyTorch.

    """

    questions = [

        "How many pretrained models are available in 🤗 Transformers?",

        "What does 🤗 Transformers provide?",

        "🤗 Transformers provides interoperability between which frameworks?",

    ]

    return text, questions


def main():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # bert-base-cased-finetuned-mrpc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    text, questions = prepare_task()

    for question in questions:
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")

        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)

        answer_start_scores = outputs.start_logits

        answer_end_scores = outputs.end_logits

        # Get the most likely beginning of answer with the argmax of the score

        answer_start = torch.argmax(answer_start_scores)

        # Get the most likely end of answer with the argmax of the score

        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(

            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])

        )

        print(f"Question: {question}")

        print(f"Answer: {answer}")


if __name__ == '__main__':
    main()
