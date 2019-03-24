import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

pylab.rcParams['figure.figsize'] = 20, 12

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    

if __name__ == '__main__':
#    image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
    import cv2
    image = cv2.imread("/home/lyh/dataset/lsp_dataset_original/images/im0021.jpg")
   # imshow(image)
    
    
    #config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    #config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_C4_1x_caffe2.yaml"
    #config_file = "../configs/caffe2/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml"
    config_file = "../configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml"
    
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    
    
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
    )

    predictions = coco_demo.run_on_opencv_image(image)

    imshow(predictions)

    cv2.imshow("result", predictions)
    cv2.waitKey()
