import torch
from PIL import Image
from torchvision import transforms

# image
def load_image(filename, size=None):
    image = Image.open(filename).convert('RGB')
    if size is not None:
        image = image.resize((size, size), Image.ANTIALIAS)

    return image

def save_image(filename, data):
    img = data.detach().clamp(0, 255).numpy()
    img = img.transpose(1,2,0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def get_image_tensor(image_path, size=None):
    image = load_image(image_path, size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = transform(image)
    return image


# use imagenet to initialize before into VGG
def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)

    return (batch - mean) / std


# get gram matrix
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)

    return gram
