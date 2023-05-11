import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

class FasterRCNN(BaseModel):
    """FasterRCNN_fpn custom module
     .. note::
        load pretrained model
    """
    def __init__(self, num_classes:int):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.num_classes = num_classes # class 개수= 10 + background

        # get number of input features for the classifier
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)

    def forward(self, images, targets):
        """
        Args:
            images(images) : Different images can have different sizes. and expected 0~1 range
            targets(list of dictionary)
                - boxes  : [x1, y1, x2, y2] 
                - labels : the class label for each ground-truth box
        """
        net = self.model(images, targets)
        return net