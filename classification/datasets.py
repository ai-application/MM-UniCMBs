# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from torch.utils.data import dataset
import torchvision.transforms as transforms
from bert_embedding import BertEmbedding
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CMB_Dataset(dataset.Dataset):
    def __init__(self, args, dataset_path, flag, dataset_name, batch_size):
        super(CMB_Dataset, self).__init__()

        self.batch_size = batch_size

        self.flag = flag

        self.dataset_name = dataset_name

        self.path = dataset_path

        self.slices = 3

        self.input_size = args.input_size

        self.bert_encoding = BertEmbedding()

        self.classes = 0

        self.text_input = args.input_text

        self.use_bert = args.use_bert

        self.use_clip_bedding = args.use_clip_image

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model, self.image_preprocess = clip.load("ViT-B/32", device="cpu")

        if not os.path.exists(self.path):
            print("############################################")
            print(self.path)
            print("############################################")
            raise ValueError

        self.transform = {

        'train': transforms.Compose([
            transforms.Resize([self.input_size, self.input_size]),  # h, w

            transforms.Grayscale(3),

            transforms.ToTensor(),

            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        ]),

        'valid': transforms.Compose([

            transforms.Resize([self.input_size, self.input_size]),

            transforms.Grayscale(3),

            transforms.ToTensor(),

            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

            ])
        }

        cls_dirs = []
        for root, dirs, files in os.walk(self.path):
            cls_dirs = dirs
            break

        self.classes = len(cls_dirs)

        self.imgs = []
        self.labels = []

        if flag == 'train':
            print("##################################################")
            print("The classification labels indexs lists is: {0}".format(cls_dirs))
            print("##################################################")

            with open(self.dataset_name + "_class_labels_name.txt", "w") as f:
                class_info = "["
                for name in cls_dirs:
                    class_info = class_info + str('\'') + name + str('\'')+ ", "

                class_info = class_info[0:len(class_info) - 2] + "]"
                print(class_info)
                f.write(class_info)

        for index, class_path in enumerate(cls_dirs):
            cur_class_path = os.path.join(self.path, class_path)
            class_dirs = []
            for root, dirs, files in os.walk(os.path.join(cur_class_path)):
                class_dirs = dirs
                break

            self.imgs.extend([os.path.join(cur_class_path, dir) for dir in class_dirs])
            self.labels.extend([index for i in range(0, len(class_dirs))])

    def __getitem__(self, index):

        image_tensors = []
        images = []

        cur_img_path = os.path.join(self.path, self.imgs[index])

        files = os.listdir(cur_img_path)

        for i in range(0, len(files)):
            if files[i].endswith('.jpg'):
                pil_img = Image.open(os.path.join(cur_img_path, files[i]))
                images.append(pil_img)

                if self.flag == 'train':
                    src_tensor = self.transform['train'](pil_img)
                else:
                    src_tensor = self.transform['valid'](pil_img)

                image_tensors.append(src_tensor)

        img_tensor = torch.stack(image_tensors)

        img_tensor = img_tensor.permute(1, 0, 2, 3)

        clip_im_tensor = torch.zeros([1, 1], dtype=torch.float)

        if self.text_input:
            texts = []
            with open(os.path.join(cur_img_path, 'description.txt'), "r") as f:
                for line in f.readlines():
                    if len(line.rstrip('\n')) > 0:
                        texts.append(line.rstrip('\n'))

            with torch.no_grad():
                if self.use_bert:
                    text_tokens = self.bert_encoding(texts)

                    text_encoding = torch.tensor(np.concatenate([np.array(text_tokens[i][1]) for i in range(0, len(text_tokens))], axis = 0))
                else:
                    text_encoding = self.clip_model.encode_text(clip.tokenize(texts))

                if self.use_clip_bedding:
                    clip_tensors = []

                    for i in range(0, len(images)):
                            pro_image = self.image_preprocess(images[i]).unsqueeze(0)

                            image_features = self.clip_model.encode_image(pro_image).squeeze()

                            clip_tensors.append(image_features)

                    clip_im_tensor = torch.stack(clip_tensors)

            return img_tensor, self.labels[index], text_encoding, clip_im_tensor

        else:
            return img_tensor, self.labels[index], torch.zeros([1,1], dtype=torch.float), clip_im_tensor

    def __len__(self):
        return len(self.imgs)

    def get_class_nums(self):
        return self.classes
