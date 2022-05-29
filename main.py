import argparse
from train import *
import numpy as np
import pandas as pd
import warnings
import cv2
import mediapipe as mp
import time
import os
from PIL import Image

warnings.filterwarnings(action='ignore')
## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=120, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./../datasets/img_align_celeba", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="DCGAN", choices=["DCGAN"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4], dest='opts')

parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=128, type=int, dest="nker")

parser.add_argument("--network", default="DCGAN", choices=["unet", "hourglass", "resnet", "srresnet", "DCGAN"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()


DATA_FOLDER = "C://Users/user/PycharmProjects/datasets" #데이터들 위치지정
TRAIN_SAMPLE_FOLDER = 'train_videos' #훈련용 폴더이름
TEST_FOLDER = 'test_videos' #테스트용 폴더이름
REAL_FOLDER = 'real_picture' #테스트용 폴더이름
FAKE_FOLDER = 'fake_picture' #테스트용 폴더이름


#print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")

#print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER))) #훈련 파일들 이름 리스트 train_list

json_file = [file for file in train_list if  file.endswith('json')][0] #json파일 찾기
#print(f"JSON file: {json_file}")#json 파일 이름 찾기

def get_meta_from_json(path): #json 읽기
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER) #train json 읽기
meta_train_df.head()

def missing_data(data): #json과 실제 비교
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) #가로로 합치기, 이름 붙이기

    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

missing_data(meta_train_df)
missing_data(meta_train_df.loc[meta_train_df.label=='REAL'])

meta = np.array(list(meta_train_df.index))
storage = np.array([file for file in train_list if  file.endswith('mp4')])
#print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")
#print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")#metadata에 있는데 실제로 없는 데이터
#print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")#실제로 있는데 metadata에 없는 데이터
#print(np.setdiff1d(meta,storage,assume_unique=False))

imgNum = 0
for i in train_list:
    name = i
    cap = cv2.VideoCapture(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, name)) # 캡쳐해서

    mpFaceDetection = mp.solutions.face_detection#디텍션도구세팅
    mpDraw = mp.solutions.drawing_utils

    faceDetection = mpFaceDetection.FaceDetection()  # 페이스디텍션 시작
    print(meta_train_df.loc[[name]])
    if not os.path.exists(os.path.join(DATA_FOLDER, REAL_FOLDER)):
        os.makedirs(os.path.join(DATA_FOLDER, REAL_FOLDER))
    if not os.path.exists(os.path.join(DATA_FOLDER, FAKE_FOLDER)):
        os.makedirs(os.path.join(DATA_FOLDER, FAKE_FOLDER))
    tf=1
    while True:
        try:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB)
            results = faceDetection.process(imgRGB)
        except:
            break

        if results.detections:  # 가능하면
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img,detection)
                # print(id,detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
                # 이미지를 저장
#                name.rstrip('.mp4')
                try:
                    if(meta_train_df.loc[name].label=='REAL'):
                        cv2.imwrite(DATA_FOLDER + "/" + REAL_FOLDER + "/" + str(imgNum) + ".png", cropped)
                        print(str(imgNum)+"번 사진이 REAL 폴더에 저장됨")
                        img = Image.open(DATA_FOLDER + "/" + REAL_FOLDER + "/" + str(imgNum) + ".png")
                        img_resize = img.resize(1024, 1024)
                        img_resize_lanczos.save(DATA_FOLDER + "/" + REAL_FOLDER + "/" + str(imgNum) + ".png")

                    elif(meta_train_df.loc[name].label=='FAKE'):
                        cv2.imwrite(DATA_FOLDER + "/" + FAKE_FOLDER + "/" + str(imgNum) + ".png", cropped)
                        print(str(imgNum)+"번 사진이 FAKE 폴더에 저장됨")
                except:
                    pass
                imgNum += 1






if __name__ == "__main__":
    if args. mode == "train":
        pass#train(args)
    elif args.mode == "test":
        pass#test(args)

# tensorboard --logdir ./log
