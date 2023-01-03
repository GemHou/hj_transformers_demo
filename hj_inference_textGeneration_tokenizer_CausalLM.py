from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate_results(model, sequence1, sequence2, tokenizer):
    paraphrase = tokenizer(sequence1, sequence2, return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase).logits
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    return paraphrase_results


def prepare_task():
    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family

    (except for Alexei and Maria) are discovered.

    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the

    remainder of the story. 1883 Western Siberia,

    a young Grigori Rasputin is asked by his father and a group of men to perform magic.

    Rasputin has a vision and denounces one of the men as a horse thief. Although his

    father initially slaps him for making such an accusation, Rasputin watches as the

    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of

    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,

    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    prompt = "Today the weather is really nice and I am planning on "

    return PADDING_TEXT, prompt


def main():
    model_name = "xlnet-base-cased"  # bert-base-cased-finetuned-mrpc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    PADDING_TEXT, prompt = prepare_task()

    inputs = tokenizer(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]  # PADDING_TEXT +

    prompt_length = len(tokenizer.decode(inputs[0]))
    # input_ids = inputs["input_ids"].tolist()[0]

    # outputs = model(**inputs)
    outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)

    generated = prompt + tokenizer.decode(outputs[0])[prompt_length + 1:]

    print("generated: ", generated)


if __name__ == '__main__':
    main()
