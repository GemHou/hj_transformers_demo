from transformers import pipeline


def main():
    print("hello world...")
    classifier = pipeline('sentiment-analysis')
    returns = classifier('We are very glad to introduce pipeline to the transformers repository.')
    print("returns: ", returns)


if __name__ == '__main__':
    main()
