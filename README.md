# AerialObjectDetection_FasterRCNN_YOLOv5_on_DOTA

This project aims to improve the accuracy and efficiency of object detection models for aerial imagery by re-implementing existing models and developing new processes. We will address the unique challenges of aerial imagery, such as variable lighting, scale, orientation, and spatial resolution.

Our approach involves creating an end-to-end pipeline for aerial object detection, from data preprocessing to model training and testing. We will evaluate state-of-the-art models, such as YOLO, Faster R-CNN, and SSD, and explore techniques like data augmentation, transfer learning, and region proposal methods.

We will test the performance of the system on publicly available aerial imagery datasets. The project aims to produce a guide for implementing object detection models and processes for aerial imagery and an open-source code repository for the proposed pipeline.

The primary goal is to create a more accurate and efficient system for object detection in aerial imagery, which will have various applications in the domain of remote sensing and beyond. We will measure the system's performance in terms of mAP, precision, recall, F1 score, and detection speed.

Successfully optimized deep learning models to detect 15 distinct objects through implementation of image tiling and innovative Strategic Aerial Homogenization for Inference (SAHI) approach to improve mean average precision by 36% 

Detecting small objects has always been a problem with Aerial Object detection. To overcome this, we have discussed about adding a preprocessing step of tiling the images and use of a plug-in kind of method called SAHI which enhances the detection for a given image by mainly slicing and iterating the image.

The fine-tuned model when used with SAHI framework for Inference provides better detection since it uses slicing even in its inference step.

DATASET
For this project, we have used DOTA Dataset. DOTA is a large-scale dataset for object detection in aerial images. It can be used to develop and evaluate object detectors in aerial images. The images are collected from different sensors and platforms. Each image is of the size in the range from 800 × 800 to 20,000 × 20,000 pixels and contains objects exhibiting a wide variety of scales, orientations, and shapes. There are many versions of DOTA (v1.0, v1.5, v2.0). Here we have used DOTA v1.0, it has the following object categories, plane, ship, storage tank, baseball diamond, tenniscourt, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccerball field and swimming pool.
One of the major problems with DOTA is that we cannot use it directly for training, especially on object detection tasks. Most of the state-of-the-art models rely on different annotations formats. The most popular are:
●	COCO — a single JSON file consists of five sections of information for the entire datasets.
●	YOLO — individual text file per image with the same name corresponding to the intended image.
So, we have converted the DOTA dataset to COCO format. We used DOTA Devkit tool for splitting and tiling of images.

MODELS:
YOLOv5
FASTER-RCNN

Perfomrance Metrics:
mAP, Precision, Recall, F1 Score, 

Techniques Used: 
1.	Tiling: In order to alleviate small object problems, we decrease the effect of image down-sampling that is a common tool applied during training. The images are divided into smaller images by the help of overlapping tiles, where the sizes of tiles are 1024x1024 and 512x512. Each tile corresponds to a new image where ground truth object locations are arranged accordingly without change of object size. In that way, the relative object sizes are increased in the cropped image compared to the full frame. The overlaps between the tiles are used to preserve the objects along the tile boundaries and prevent any miss due to image partitioning. In this study, we have chosen 200 as the intersection between consecutive tiles in case of 1024x1024 images and as 64 pixels in case of 512x512 tiles. Tiling provides a more detailed look at specific regions of the images. The area occupied by the smaller objects appears larger when the tiling is done. As the tiling increases, larger objects may not fit within the tiles and the intersecting areas, and the risk of loosing larger object annotations also increases. Therefore, at some point the increase in tiles starts to decrease the number of annotations. An example of tiling is shown below. The original image on the left is of size 3875x5502 size. The red highlighted section of the image is split into different tiles as shown in the right. 
 
2.	Augmentations: We have applied different types of augmentations into the YOLOV5 models as given below:
a.	HSV augmentation: The hue, saturation and value of the images were tweaked to prevent the problem of illumination in the images. The values used are:
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
b.	Translation: It means moving the image along the X or Y direction. The value taken is translate: 0.1  # image translation (+/- fraction)
c.	Scale: It means randomly pick the short size of a image within a dimension range. The value used is scale: 0.5  # image scale (+/- gain).
d.	Flip left-right : It helps us to deal with the orientation and size problems. The values used are fliplr: 0.5  # image flip left-right (probability).
e.	Mosaic: It takes 4 images and combines them into a single image. This helps us to deal with occlusion problems. The values used are mosaic: 1.0  # image mosaic (probability)
3.	SAHI: (Slicing Aided Hyper Inference) The concept of sliced inference is basically; performing inference over smaller slices of the original image and then merging the sliced predictions on the original image. Small object detection is thus a difficult task in computer vision because, in addition to the small representations of objects, the diversity of input images makes the task more difficult. To address the small object detection difficulty, Fatih Akyon et al. presented Slicing Aided Hyper Inference (SAHI), an open-source solution that provides a generic slicing aided inference and fine-tuning process for small object recognition. During the fine-tuning and inference stages, a slicing-based architecture is used. [4]
We have used SAHI during the inference step i.e., we have fine-tuned the yolov5 model with our custom dataset (DOTA v1.0) then used the same model along with SAHI during the inference process. This is possible because even during the inference step, the slicing method is also used. In this case, the original query image is cut into a number of overlapping patches. The patches are then resized while the aspect ratio is kept. Following that, each overlapping patch receives its own object detection forward pass. To detect larger objects, an optional full-inference (FI) using the original image can be used. Finally, the overlapping prediction results and, if applicable, the FI results are merged back into their original size[4]. This is different than tiling because in case of tiling a single object may fall into 4 different blocks of sub-image and a single object may be detected twice. This is not the case with SAHI as the annotations or predictions are combined into one once the predictions are made on each slice of the image.
