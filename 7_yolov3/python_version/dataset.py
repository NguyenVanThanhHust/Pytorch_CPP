import os
import numpy as np
from os.path import join, isdir, isfile

import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

class WGISDMaskedDataset(Dataset):
    def __init__(self, root, transforms=None, S=7, B=2, C=2, split='train'):
        self.root = root
        self.transforms = transforms
        
        assert split in ["train", "test"], "split must be train or test, get {}".format(split)

        # source_path = os.path.join(root, f'{source}_masked.txt')
        source_path = os.path.join(root, f'{split}.txt')
        with open(source_path, 'r') as fp:
            # Read all lines in file
            lines = fp.readlines()
            # Recover the items ids, removing the \n at the end
            lines = [l.rstrip() for l in lines]


        self.img_paths = [join(self.root, "data", line+".jpg") for line in lines]
        self.label_paths = [join(self.root, "data", line+".txt") for line in lines]
        self.mask_paths = [join(self.root, "data", line+".npy") for line in lines]
        
        self.sizes = (2048, 1365)
        self.S = S
        self.B = B
        self.C = C

    def __len__(self, ):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        label_path = self.label_paths[idx]
        
        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        # img = np.transpose(img / 255., (2, 0, 1))

        with open(label_path, "r") as handle:
            lines = handle.readlines()
            lines = [l.rstrip() for l in lines]
        
        boxes = []
        for l in lines:
            _, cen_x, cen_y, _w, _h = l.split()
            cen_x, cen_y = float(cen_x), float(cen_y)
            _w, _h = float(_w), float(_h)
            # w, h = _w*self.sizes[0], _h*self.sizes[1]
            # x, y = (cen_x - _w/2)*self.sizes[0], (cen_y - _h/2)*self.sizes[1]
            w, h = _w, _h
            x, y = (cen_x - _w/2), (cen_y - _h/2)


            boxes.append([x, y, w, h])

        boxes = np.array(boxes, dtype=np.float32)
        
        labels = np.ones((boxes.shape[0], 1), dtype=np.float32)
        
        if self.transforms:
            img, boxes = self.transforms(img, boxes)
        
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for label, box in zip(labels, boxes):
            x, y, width, height = box.tolist()
            class_label = label

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C + 5 * self.B - 5] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C + 5 * self.B - 5] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, self.C + 5 * self.B - 4:self.C + 5 * self.B] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return img, label_matrix


# data = WGISDMaskedDataset(root="../../../wgisd", )
# for i in range(data.__len__()):
#     img, target = data.__getitem__(i)