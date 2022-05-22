import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
import warnings
warnings.filterwarnings(action='ignore')

DATA_FOLDER = 'C:\\Users\\SANGMAN\\input\\deepfake-detection-challenge' #데이터들 위치지정
TRAIN_SAMPLE_FOLDER = 'train_sample_videos' #훈련용 폴더이름
TEST_FOLDER = 'test_videos' #테스트용 폴더이름

print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER))) #훈련 파일들 이름 리스트

json_file = [file for file in train_list if  file.endswith('json')][0] #json파일찾기(정보)
print(f"JSON file: {json_file}")


def get_meta_from_json(path): #json 읽기
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER) #train json 읽기
meta_train_df.head()

def missing_data(data): #json과 실제 비교
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(meta_train_df)
missing_data(meta_train_df.loc[meta_train_df.label=='REAL'])

def most_frequent_values(data):
    total = data.count()#데이터 카운트
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))

print(most_frequent_values(meta_train_df))

def plot_count(feature, title, df, size=1):
    
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center")
    plt.show()

plot_count('split', 'split (train)', meta_train_df)
plot_count('label', 'label (train)', meta_train_df)



meta = np.array(list(meta_train_df.index))
storage = np.array([file for file in train_list if  file.endswith('mp4')])
print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")
print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")#metadata에 있는데 실제로 없는 데이터
print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")#실제로 있는데 metadata에 없는 데이터

fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(3).index)
print(fake_train_sample_video) #페이크비디오 리스트

def display_image_from_video(video_path): #위치의 영상 미리보기 함수

    capture_image = cv.VideoCapture(video_path)
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    ax.imshow(frame)
    plt.show()

for video_file in fake_train_sample_video:#가짜 비디오
    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))

real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(3).index)
print(real_train_sample_video) #이름 불러오기

for video_file in real_train_sample_video: # 진짜 비디오
    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))


print(meta_train_df['original'].value_counts()[0:5]) #같은 원본의 다른 비디오


test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])
test_videos.head()
display_image_from_video(os.path.join(DATA_FOLDER, TEST_FOLDER, test_videos.iloc[0].video))
