from datasets import load_dataset


def main():
    dataset = load_dataset("yelp_review_full")

    print('dataset[train][100]: ', dataset["train"][100])

    print("finished...")


if __name__ == '__main__':
    main()