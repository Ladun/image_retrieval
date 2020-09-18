import cv2
from matplotlib import pyplot as plt
import os

def main():
    with open('retrieval_result.txt', 'r') as f:
        fnames = f.readlines()

    for fname in fnames:
        path = fname[3:-1]
        print(f"[INFO] {path} read...")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()
