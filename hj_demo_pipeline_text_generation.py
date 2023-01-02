from transformers import pipeline


def main():
    print("hello world...")
    text_generator = pipeline("text-generation")
    results = text_generator('We are very glad to introduce pipeline to the transformers repository.')[0]
    print("results: ", results)
    results = text_generator('I hate you.')[0]
    print("results: ", results)
    results = text_generator('I love you.')[0]
    print("results: ", results)
    results = text_generator('Haha.')[0]
    print("results: ", results)
    results = text_generator('I do not think it is not okay.')[0]
    print("results: ", results)
    results = text_generator('I do think it is okay.')[0]
    print("results: ", results)


if __name__ == '__main__':
    main()
