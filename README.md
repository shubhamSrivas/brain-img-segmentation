# Brain MRI Segmentation #
This project aims at detecting superficial brain lesions, such as tumor, hematoma, abscess, cyst, or cavernoma on MRI images. 

Dataset Link :- https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

Original Unet-Architecture :- https://arxiv.org/abs/1505.04597

## Description ##
Module called ImageDataGenerator in keras.preprocessing.image was used to do data augmentation.

It employs Unet-based Architecture to generate masks of corresponding MRI images given in RGB format. 

The loss function used was Dice loss. 

The framework used was keras.

## Result ##
![](https://github.com/Shashwat07gupta/Unet_brain_MRI_segmentation/blob/master/media/img_3.jpeg)

MRI image with brain lesion

![](https://github.com/Shashwat07gupta/Unet_brain_MRI_segmentation/blob/master/media/img_2.jpeg)

MRI image without brain lesion
