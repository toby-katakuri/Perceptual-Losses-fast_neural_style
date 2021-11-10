import argparse
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from transformer_net import TransformerNet
from vgg import Vgg16
import utils
import time
import os
from tqdm import tqdm
import numpy as np


## argument config##

eval_arg_parser = argparse.ArgumentParser()

# path
# content
eval_arg_parser.add_argument("--content_image", type=str, default='data/content_image/ying.jfif')

eval_arg_parser.add_argument("--output-image", type=str, default='output')
eval_arg_parser.add_argument("--save-model-dir", type=str, default='models')
eval_arg_parser.add_argument("--model", type=str, default='output')
eval_arg_parser.add_argument("--cuda", type=int, default=1)

# model
eval_arg_parser.add_argument("--model_name", type=str, default='epoch_1_02.pt')

# image
eval_arg_parser.add_argument('--image_size', type=int, default=256*2)

args = eval_arg_parser.parse_args()

if __name__ == '__main__':
    print('-----Eval Mode-----')
    device = torch.device('cuda' if args.cuda else 'cup')

    # load content image
    content_image = utils.get_image_tensor(args.content_image, args.image_size)

    # show origin image
    content_filename = os.path.join(args.output_image, args.model_name[:-3] + '_' + os.path.basename(args.content_image)[:-4] +
                                    '_origin' + '.jpg')
    utils.save_image(content_filename, content_image)

    # get content image for eval
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(os.path.join(args.save_model_dir, args.model_name))
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()

    # show stylized image
    output_filename = os.path.join(args.output_image, args.model_name[:-3] + '_' + os.path.basename(args.content_image)[:-4] +
                                   '_style' + '.jpg')
    utils.save_image(output_filename, output[0])
