import numpy as np

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


from guide.feature_extractor import preprocess
from guide.image_retrieval import ImageRetrieval


def retrieval_res(image_path, store_file):

    binary_file = f"{store_file}.bin"
    name_file = f"{store_file}_names.txt"
    # query Image feature 뽑는 과정
    # 현재는 간단하게 Resnet152로 이미지의 feature를 뽑아내서 비교
    dim = 2048
    input_shape = (224, 224, 3)
    base = tf.keras.applications.ResNet152(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False
    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    # 이미지 로드
    img = preprocess(image_path, input_shape)

    # 이미지에서 feature 뽑아내기
    fvec = model.predict(np.array([img]))

    # 이미지 검색 클래스 생성
    # fvecs.bin랑 fnames.txt는 feature_extractor.py에서 만들 수 있는 파일
    imageRetrieval = ImageRetrieval(fvec_file=binary_file,
                                    fvec_img_file_name=name_file,
                                    fvec_dim=dim)

    results = imageRetrieval.search(fvec)

    with open('retrieval_result.txt', 'w') as f:
        for path, dst in results:
            print(f"Image path: {path}, dst: {dst}")
            f.writelines(f"{path}\n")


def retrieval_delf(image_path, store_file):
    def get_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    binary_file = f"{store_file}.bin"
    name_file = f"{store_file}_names.txt"

    extract_size = 300
    dim = 40 * 300

    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    img = get_image(image_path)

    delf_rt = delf(image=img,
                   score_threshold=tf.constant(100.0),
                   image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
                   max_feature_num=tf.constant(1000))

    descrip = delf_rt['descriptors']
    descrip = np.array(descrip)
    if descrip.shape[0] < extract_size:
        print(f"[INFO] {image_path} feature size: {descrip.shape}")
        return
    descrip = descrip[:extract_size]
    descrip = descrip.flatten().reshape(1, -1)

    imageRetrieval = ImageRetrieval(fvec_file=binary_file,
                                    fvec_img_file_name=name_file,
                                    fvec_dim=dim)

    results = imageRetrieval.search(descrip)
    with open('retrieval_result.txt', 'w') as f:
        for path, dst in results:
            print(f"Image path: {path}, dst: {dst}")
            f.writelines(f"{path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                        help="Query Image")
    parser.add_argument("--store", required=False,
                        default="fvecs",
                        help='Features Binary File')
    parser.add_argument("--type", required=False,
                        default="res",
                        help='Extract type ( "res" or "delf" )')
    args = parser.parse_args()

    if args.type == "res":
        retrieval_res(args.image, args.store)
    elif args.type == "delf":
        retrieval_delf(args.image, args.store)
