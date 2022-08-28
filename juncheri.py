import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#image import
from PIL import Image
path = 'C:\\Users\\SANGMAN\\PycharmProjects\\junram\\smalldata'
for i in range(310, 410):
  fpath=path+'\\081'+str(i)+'_A128by.jpg'
  img_array = np.fromfile(fpath, np.uint8)
  globals()['image{}'.format(i)] = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 이미지를 읽어옴

#PIL_image=Image.fromarray(image400)
#PIL_image.show()

for i in range(310, 410):
  globals()['image{}'.format(i)] = locals()['image{}'.format(i)].reshape(-1,16384)

data=np.ones(16384)
for i in range(310, 410):
  data = np.vstack((data,locals()['image{}'.format(i)]))
data_saved=pd.DataFrame(data, columns=np.arange(0,16384))
data_saved.to_csv("data.csv")

input = data_saved[np.arange(0,16384)]
print(input)
target=data_saved[100]
print(target)