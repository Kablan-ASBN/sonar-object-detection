import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

class VOCDataset(Dataset):
    def __init__(self, root, image_set="train", transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")

        with open(os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")) as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        self.class_map = {"object": 1, "object_alt": 2}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        ann_path = os.path.join(self.ann_dir, img_id + ".xml")

        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(ann_path)

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            label = self.class_map.get(name, 1)

            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            # Filter invalid boxes: width or height < 1 pixel
            if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if len(boxes) == 0:
            # Add dummy box to avoid crash
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # dummy background label

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return target