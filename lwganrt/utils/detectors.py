import torch
import torchvision

from lwganrt.utils.util import morph


class PersonMaskRCNNDetector(object):
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    PERSON_IDS = 1

    def __init__(self, ks=3, threshold=0.5, device=None, pretrained_path=None):
        super(PersonMaskRCNNDetector, self).__init__()

        pretrained = True if pretrained_path is None else False
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained)

        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))

        self.model.eval()

        self.threshold = threshold
        self.ks = ks
        self.kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32)

        if device is None:
            self.model = self.model.cuda()
            self.kernel = self.kernel.cuda()
        else:
            self.model = self.model.to(device)
            self.kernel = self.kernel.to(device)

    def forward(self, images):
        predictions = self.model(images)
        return predictions

    def get_bbox_max_ids(self, labels, bboxs):
        """
        Args:
            labels:
            bboxs: [N, 4], [x0, y0, x1, y1]

        Returns:

        """

        cur_pid = -1
        cur_bbox_area = -1
        for i, label in enumerate(labels):
            if label == self.PERSON_IDS:
                x0, y0, x1, y1 = bboxs[i]
                cur_area = torch.abs((x1 - x0) * (y1 - y0))

                if cur_area > cur_bbox_area:
                    cur_bbox_area = cur_area
                    cur_pid = i

        return cur_pid

    def inference(self, img):
        img_list = [(img + 1) / 2.0]

        with torch.no_grad():
            predictions = self.forward(img_list)[0]
            labels = predictions['labels']
            bboxs = predictions['boxes']
            masks = predictions['masks']

            pid = self.get_bbox_max_ids(labels, bboxs)

            pid_bboxs = bboxs[pid]
            pid_masks = masks[pid]

            final_masks = (pid_masks > self.threshold).float()

            if self.ks > 0:
                final_masks = morph(final_masks[None], self.ks, mode='dilate', kernel=self.kernel)

            return pid_bboxs, final_masks








