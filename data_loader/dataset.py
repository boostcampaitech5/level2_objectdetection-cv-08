from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

import pandas as pd
from tqdm import tqdm

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from utils import *
from utils.BoundingBox import BoundingBox
from utils.BoundingBoxes import BoundingBoxes
from utils.Evaluator import *

class CustomDataset(Dataset):
    '''
    CustomDataset
    .. note
        Pascal VOC 포맷으로 제출할 것
        한 row : image_id / PredictionString
        PredictionString = label score xmin ymin xmax ymax
        기존 코드는 CV stratege가 없음
    '''

    def __init__(self, annotation:str, data_dir:str, transforms=None,indices=None):
        """
        Args:
            annotation : abspath of train.json file
            data_dir   : abspath of dataset file
            transforms : transform
            indices    : index of image for split train / valid

        return:
            image      : image 
            target     : target info(dictionary)
                {'boxes': boxes, 'labels': labels, 
                'image_id': torch.tensor([index]), 'area': areas,
                'iscrowd': is_crowds}
            image_id   : coco image id in json file
        """

        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.annotation = annotation #annotation path
        self.coco = COCO(self.annotation) #annotation : train.json
        
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms
        
        # train과 validation indices를 저장하는 배열 / setup실행으로 나누어줌
        self.indices = indices

    def __getitem__(self, index: int):        
        index = self.indices[index]
            
        # coco는 train.json 파일을 가져와서 파싱을 해준다.
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        
        # 파싱으로 가져온 경로를 붙혀서 imread로 이미지를 실제로 하나씩 가져온다.
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        #255로 나눠줌을로써 정규화를 진행
        # torchvision faster r cnn은 입력으로 들어오는 데이터가 0~1사이
        image /= 255.0

        # ann : 박스에 관한 정보가 담겨있는 듯
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # boxes : (xmin, ymin, w, h)
        # coco는 (좌상단 x, 좌상단 y, w, h)
        boxes = np.array([x['bbox'] for x in anns])

        # boxes (x_min, y_min, x_max, y_max)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # torchvision faster_rcnn은 label=0을 background로 취급
        # class_id를 1~10으로 수정
        # 해당 데이터는 배경 레이블링이 진행되지 않았음
        labels = np.array([x['category_id']+1 for x in anns]) 
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 박스의 넓이
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
                     
        # 한 이미지내에 많은 객체가 포함되어 있나?
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas,
                  'iscrowd': is_crowds}

        # transform
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.indices)
        