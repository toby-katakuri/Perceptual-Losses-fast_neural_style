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


## argument config ##

train_arg_parser = argparse.ArgumentParser()
# train config
train_arg_parser.add_argument("--model_name", type=str, default='02')
train_arg_parser.add_argument("--epochs", type=int, default=1)
train_arg_parser.add_argument("--batch_size", type=int, default=4)
train_arg_parser.add_argument("--lr", type=float, default=1e-3)

# path
train_arg_parser.add_argument("--dataset", type=str, default=r'E:\datasets\train2014')
train_arg_parser.add_argument("--style-image", type=str, default="data/style_image/0026.jpg")
train_arg_parser.add_argument("--content_image", type=str, default='data/content_image/scenery02.jpg')
# save model
train_arg_parser.add_argument("--save-model-dir", type=str, default='models')

# image size
train_arg_parser.add_argument("--image-size", type=int, default=256)
train_arg_parser.add_argument("--style-size", type=int, default=256)
train_arg_parser.add_argument("--cuda", type=int, default=1)
train_arg_parser.add_argument("--seed", type=int, default=42)

# weight
train_arg_parser.add_argument("--content-weight", type=float, default=1e5)
train_arg_parser.add_argument("--style-weight", type=float, default=1e10)
train_arg_parser.add_argument("--log-interval", type=int, default=500)
train_arg_parser.add_argument("--checkpoint-interval", type=int, default=1000)

args = train_arg_parser.parse_args()


## train ##
if __name__ == '__main__':

    print('-----Train Mode-----')
    device = torch.device('cuda' if args.cuda else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load content data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))

    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # create Net
    transformer = TransformerNet().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    # load style image
    style = utils.get_image_tensor(args.style_image, args.style_size)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    # style into vgg to get gram
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # ready to train
    torch.backends.cudnn.benchmark = True
    for e in range(args.epochs):
        print('Epoch: {}'.format(e),sep='\n')
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0.
        for batch_id, (x, _) in enumerate(tqdm(train_loader)):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            # start training
            # x is tensor,0-255
            x = x.to(device)
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)
            style_loss = torch.tensor([0.], device=device)
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            # show loss and checkpoint image
            if (batch_id + 1) % args.checkpoint_interval == 0:
                # show loss
                mesg = '{}\tEpoch {}: \t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}'.format(
                    time.ctime(), e+1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

                # show picture
                check_image = utils.get_image_tensor(args.content_image, args.image_size)
                check_image = check_image.unsqueeze(0).to(device)
                output = transformer(check_image).cpu()
                utils.save_image('checkpoint/'+'{}.jpg'.format(batch_id + 1), output[0])

            # show origin image
            if e == 0 and batch_id == 0:
                check_image0 = utils.get_image_tensor(args.content_image, args.image_size)
                utils.save_image('checkpoint/' + '{}.jpg'.format(1), check_image0)

    transformer.eval().cpu()
    # save model
    save_model_filename = "epoch_" + str(args.epochs) + '_' + args.model_name + ".pt"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)