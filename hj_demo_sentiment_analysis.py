from transformers import pipeline


def main():
    print("hello world...")
    classifier = pipeline('sentiment-analysis')
    results = classifier('We are very glad to introduce pipeline to the transformers repository.')[0]
    print("results: ", results)
    results = classifier('I hate you.')[0]
    print("results: ", results)
    results = classifier('I love you.')[0]
    print("results: ", results)
    results = classifier('Haha.')[0]
    print("results: ", results)
    results = classifier('I do not think it is not okay.')[0]
    print("results: ", results)
    results = classifier('I do think it is okay.')[0]
    print("results: ", results)


if __name__ == '__main__':
    main()
