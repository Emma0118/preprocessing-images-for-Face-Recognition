this is an integrated implementation of preprocessing image data for Face Recognition, including aligning image using MTCNN, generating image and 
corresponding label list, etc.




# How to use it

## MTCNN: get aligned images with customized output size by MTCNN
_`pytorch` implementation of **inference stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)._

    1. Download the repository

    2. to get bounding_boxes and landmarks only:

```python
from src import detect_faces
from PIL import Image

image = Image.open('image.jpg')
bounding_boxes, landmarks = detect_faces(image)
```

    3. to get aligned images in folds:
````python 
python get_aligned_face_from_mtcnn.py -ops=<the output image size that you want, you must input a tuple>
````

For examples see `get_aligned_face_from_mtcnn.py`

    4. for instruction of how to use MTCNN step by step, please refer to `try_mtcnn_step_by_step.ipynb`
    
## generate image label list
in most cases, we get large training and test data, which needs a corresponding label list of int type from 0 to len(dataset), such as 
```text
3142219/002.jpg 0
3142219/003.jpg 0
3142219/001.jpg 0
3142219/004.jpg 0
3142219/010.jpg 0
3142219/011.jpg 0
3142219/005.jpg 0
3142219/007.jpg 0
3142219/006.jpg 0
3142219/008.jpg 0
3142219/009.jpg 0
3181675/016.jpg 1
3181675/003.jpg 1
3181675/017.jpg 1
3181675/029.jpg 1
3181675/001.jpg 1
3181675/015.jpg 1
```
to get such list, you run 
````python 
python get_image_label_list.py -i <your image root path> -o <your output file with its full path>
````
## More detail
    1. implementation of MTCNN
I used a pretrained MTCNN model, which is trained in Caffe. For convenience,  I have implemented code for drawing its weights for you, 
and saved them in src/weights fold.
if you like, you can get model weights yourself by running
````python 
python extract_weights_from_caffe_models.py
````
    2. detail about MTCNN algorithm and code:
* [confluence/face recognition/mtcnn](http://confluence.sensetime.com/pages/viewpage.action?pageId=64884248) 


## Future Work
this implementation of MTCNN will dectect some image with none bounding box, I think there maybe some problems here, which deserves more research.

## Requirements
* pytorch 0.4+
* Pillow, numpy