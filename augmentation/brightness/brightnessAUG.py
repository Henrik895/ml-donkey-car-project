import os
import shutil
import glob

import imageio.v2 as imageio
import imgaug as io
import imgaug.augmenters as iaa
import numpy as np

tubs = glob.glob("tub*")
range = (-50,50)
bright = iaa.color.AddToBrightness(add=range)


for tubfol in tubs:
    path = tubfol
    pathtoWrite = tubfol+"_brightAug"
    if os.path.exists(pathtoWrite+"/images"):
        print("Tub destination"+ pathtoWrite + " exist")
        break

    os.makedirs(pathtoWrite+"/images")
    print("Tub destination"+ pathtoWrite + " created")

    manifests = os.listdir(path)
    manifests.remove("images")
    [shutil.copy2(manipath,pathtoWrite) for manipath in [path+"/"+mani for mani in manifests]]

    pathImgs = tubfol+"/images"
    images = os.listdir(pathImgs)

    for img in images:
        imgPath = pathImgs+"/"+img
        imgFile  = imageio.imread(imgPath)
        augFile = bright.augment_image(imgFile)
        imageio.imwrite(pathtoWrite+"/images/"+img,augFile)