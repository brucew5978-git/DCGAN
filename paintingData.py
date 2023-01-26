import re
import urllib.request
import time
from PIL import Image
import os

results = []
targetFile = "List_of_van_Dyck_Artwork_HTML.rtf"
#HTML code file of website containing all van Dyck painting images

with open(targetFile) as file:
    for line in file: 
        match = re.search(r'src="(https://.+\.jpg)"', line)
        #Saves painting image URL in results array

        if match!=None:
            results.append(match.group(1))

imageNum = len(results)
print(imageNum)


for index in range(len(results)):
    time.sleep(0.1)
    newIndex = index+140
    urllib.request.urlretrieve(results[newIndex], f'van_Dyck_data/{newIndex}.jpg')

#Retrieves painting images from list in results array  

TARGET_FOLDER = "data/van_Dyck_data"
NEW_FOLDER = "data/van_Dyck_data_resized"

TARGET_IMG_WIDTH = 178
TARGET_IMG_HEIGHT = 218

for filename in os.listdir(TARGET_FOLDER):

    if filename.endswith(".jpg"):

        img = Image.open(os.path.join(TARGET_FOLDER, filename))

        left = (img.width/2)-(TARGET_IMG_WIDTH/2)
        upper = (img.height/2)-(TARGET_IMG_HEIGHT/2)
        right = left+TARGET_IMG_WIDTH
        lower = upper+TARGET_IMG_HEIGHT

        img = img.crop((left, upper, right, lower))

        img.save(os.path.join(NEW_FOLDER, filename))

#Resizes all images in the data folder as training image size will be 178 x 218
