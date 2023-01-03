from transformers import pipeline


def main():
    print("hello world...")
    question_answerer = pipeline('question-answering')

    context = r"""

    Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a

    question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune

    a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.

    """

    results = question_answerer(question="What is extractive question answering?", context=context)
    print("results: ", results)

    results = question_answerer(question="What is a good example of a question answering dataset?", context=context)
    print("results: ", results)

    results = question_answerer(question="What is the dataset?", context=context)
    print("results: ", results)

    results = question_answerer(question="What is the task?", context=context)
    print("results: ", results)

    results = question_answerer(question="What is the name?", context=context)
    print("results: ", results)

    results = question_answerer(question="What is the example?", context=context)
    print("results: ", results)

    results = question_answerer(question="What is the python file?", context=context)
    print("results: ", results)

    results = question_answerer(question="How to fine-tune SQuAD?", context=context)
    print("results: ", results)


if __name__ == '__main__':
    main()
