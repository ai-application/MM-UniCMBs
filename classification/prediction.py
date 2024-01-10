import os
import random
import argparse
import numpy as np
import time
import json
import torch
import time
from PIL import Image
import torchvision.transforms as transforms
from bert_embedding import BertEmbedding
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models import CmbFormer_S, CmbFormer_B, resNet3D_18, resNet3D_50, resNet3D_101, uniformer_xxs, uniformer_xs, UniFormerv2_b16, UniFormerv2_114

bert_encoding = BertEmbedding()

def get_args_parser():
    parser = argparse.ArgumentParser('CMBs Prediction script', add_help=False)

    parser.add_argument('--model-name', default='CmbFormer', type = str, help = 'model name: CmbFormer, resNet3D_50, resNet3D_101...')
    parser.add_argument('--image-path', default='test_images', type = str, help='Prediction image path')
    parser.add_argument('--nb-classes', default=2, type=int, help='class nums')
    parser.add_argument('--output-path', default='output', type=str, help='Prediction image path')
    parser.add_argument('--input-text', action="store_true", default=False, help='Texts input')
    parser.add_argument('--input-size', default=64, type=int,
                        help='Input resoution')
    parser.add_argument('--checkpoint-path', default='best.pth', type = str, help='CMBs checkpoint point path')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser

def read_data(args, dir_name):
    image_tensors = []

    cur_path = os.path.join(args.image_path, dir_name)

    files = os.listdir(cur_path)

    transform = {
        'test': transforms.Compose([
            transforms.Resize([args.input_size, args.input_size]),  # h, w

            transforms.Grayscale(3),

            transforms.ToTensor(),

            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    }

    for i in range(0, len(files)):
        if files[i].endswith('.jpg'):
            pil_img = Image.open(os.path.join(cur_path, files[i]))

            src_tensor = transform['test'](pil_img)

            image_tensors.append(src_tensor)

    img_tensor = torch.stack(image_tensors)

    img_tensor = img_tensor.permute(1, 0, 2, 3)

    if args.input_text:
        texts = []
        with open(os.path.join(cur_path, 'description.txt'), "r") as f:
            for line in f.readlines():
                if len(line.rstrip('\n')) > 0:
                    texts.append(line.rstrip('\n'))

        text_tokens = bert_encoding(texts)

        text_encoding = np.concatenate([np.array(text_tokens[i][1]) for i in range(0, len(text_tokens))], axis=0)

        return img_tensor.unsqueeze(0), torch.tensor(text_encoding).unsqueeze(0)
    else:
        return img_tensor.unsqueeze(0), torch.zeros([1, 1], dtype=torch.float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CMBs Prediction script', parents=[get_args_parser()])
    args = parser.parse_args()

    print("################################################")
    print(args)
    print("################################################")

    class_names = ['CMBs', "Not CMBs"]

    os.makedirs(args.output_path, exist_ok=True)

    if args.model_name == 'CmbFormer_S':
        model = CmbFormer_S(num_classes=args.nb_classes, text_input=args.input_text)
    elif args.model_name == 'CmbFormer_B':
        model = CmbFormer_B(num_classes=args.nb_classes, text_input=args.input_text)
    elif args.model_name == 'resNet3D_18':
        model = resNet3D_18(num_classes=args.nb_classes)
    elif args.model_name == 'resNet3D_50':
        model = resNet3D_50(num_classes=args.nb_classes)
    elif args.model_name == 'resNet3D_101':
        model = resNet3D_101(num_classes=args.nb_classes)
    elif args.model_name == 'uniformer_xxs':
        model = uniformer_xxs(num_classes=args.nb_classes)
    elif args.model_name == 'uniformer_xs':
        model = uniformer_xs(num_classes=args.nb_classes)
    elif args.model_name == 'UniFormerv2_b16':
        model = UniFormerv2_b16(num_classes=args.nb_classes)
    elif args.model_name == 'UniFormerv2_114':
        model = UniFormerv2_114(num_classes=args.nb_classes)
    else:
        raise ValueError

    print(model)

    model.eval()

    model.load_state_dict(torch.load(args.checkpoint_path)['model'], strict=False)

    model.to(args.device)

    test_dirs = []

    for root, dirs, files in os.walk(args.image_path):
        test_dirs = dirs
        break

    for dir_name in test_dirs:
        images, texts = read_data(args, dir_name)

        images = images.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)

        start = time.time()
        output = model(images, texts)
        print("Duration: {}".format(time.time() - start))

        _, pred = output.topk(1, 1, True, True)

        pred = pred.t().squeeze().cpu().numpy()

        print("Name: {}, prediction cls: {}".format(dir_name, class_names[int(pred)]))

    print("Prediction Done...")