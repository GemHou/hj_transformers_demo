import requests
from PIL import Image
from transformers import pipeline


def main():
    print("hello world...")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    image_data = requests.get(url, stream=True).raw
    image = Image.open(image_data)

    object_detector = pipeline('object-detection')
    returns = object_detector(image)
    print("returns: ", returns)


if __name__ == '__main__':
    main()
