
# import libraries
import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pandas as pd

# disable specific warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
img = cv2.imread('Import image path here')

arr = input('select the language: ') # input language code (ex: en, id, ja, ko, zh, etc)
language = list(map(str, arr.split(' ')))
print("please wait...")

reader = easyocr.Reader(language, gpu=False, verbose=False)
result = reader.readtext(img)
# print(result)

df = pd.DataFrame(result, columns=['Bounding Box', 'Text', 'Times'], dtype=object)
df['Text'] = df['Text'].astype(str)
print(df)

count = 0
for detection in result:
    count += 1
    # print(count)

    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
 
    text = detection[1]
    font = ImageFont.truetype("arial.ttf", 13) #change font if apply another language (ex: arial.ttf, arialbd.ttf, ariali.ttf, arialbi.ttf, etc)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
 
    draw.rectangle([top_left, bottom_right], outline=(0, 255, 0) , width = 2)
    # draw.text((top_left[0], top_left[1] - 18), text, font = font, fill=(0, 255, 0))
    draw.text((top_left[0], top_left[1] - 18), str(count) + ') ' + text, font = font, fill=(0, 255, 0))

    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # img = cv2.rectangle(img,top_left, bottom_right, (0,255,0), 0) 
    # img = cv2.putText(img, text, top_left, font, 0.7, (0,255,0), 2, cv2.LINE_AA)

plt.imshow(img)
plt.show()

