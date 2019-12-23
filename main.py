#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp
import pdb
import click
import cv2
import matplotlib.cm as cm
from os import makedirs
from os.path import isdir, isfile
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from sta import *
from opda import *
from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(topk):
        # In this example, we specify the high confidence classes
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo2(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

def unset_training(model):
    for item in model.parameters():
        item.requires_grad = False


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
@click.option("--num-class","-n", type=int, multiple=True, required = True)
@click.option("-m","--da-model", type=str, required = True)
def demo3(image_paths, topk, output_dir, cuda, num_class, da_model):
    """
    Generate Grad-CAM with original models
    """
    device = get_device(cuda)

    # Synset words
    #classes = get_classtable()

    # Third-party model from my other repository, e.g. Xception v1 ported from Keras
    #model = torch.hub.load(
    #    "kazuto1011/pytorch-ported-models", "xception_v1", pretrained=True
    #)

    #target_images = ['mediumresidential01','mediumresidential03', 'mediumresidential14',\
    #        'mediumresidential21', 'mediumresidential36','mediumresidential37'] + \
    #['mediumresidential38','mediumresidential39', 'mediumresidential40',\
    #        'mediumresidential41', 'mediumresidential43','mediumresidential44']
    num_class, = num_class
    #target_images = ['baseballdiamond96', 'airplane18', 'runway42', 'sparseresidential77', 'mediumresidential21']
    target_images = ['runway00', 'runway02', 'runway03', 'runway04', 'runway05', 'runway06', 'runway07', 'runway08']
    classes = [str(i) for i in range(num_class)]
    if da_model == 'STA':
        feature_extractor = ResNetFc(model_name='resnet50',model_path='/home/at7133/Research/Domain_adaptation/Separate_to_Adapt/resnet50.pth')
        cls = CLS(feature_extractor.output_num(), num_class, bottle_neck_dim=256)
        model = nn.Sequential(feature_extractor, cls).cuda()
        model.load_state_dict(torch.load('/home/at7133/Research/Domain_adaptation/Separate_to_Adapt/Only_source_classifier.pth'))
    elif da_model == 'OPDA':
        G, C = get_model('vgg', num_class=num_class, unit_size=1000)
        load_model(G, C, '/home/at7133/Research/Domain_adaptation/OPDA_BP/checkpoint/checkpoint_99')
        model = nn.Sequential(G, C).cuda()
    model.to(device)
    #unset_training(model)
    model.eval()
    # Check available layer names
    print("Layers:")
    for m in model.named_modules():
        print("\t", m[0])

    # Here we choose the last convolution layer
    #target_layer = "exit_flow.conv4"
    if da_model == 'STA':
        target_layer = "0.model_resnet.layer4.2.conv3"
    elif da_model == 'OPDA':
        target_layer = "0.lower.36" #TODO find proper target layer

    # Preprocessing
    def _Normalize(img, mean, std):
        if isinstance(img, torch.FloatTensor):
            mean = torch.FloatTensor(mean)
            std = torch.FloatTensor(std)
        else:
            raise TypeError(f'Expected Torch floattensor, got {type(img)}')
        return (img - mean)/std
    def _preprocess(image_path, img_shape):
        raw_image = cv2.imread(image_path)
        #raw_image = cv2.resize(raw_image, model.image_shape)
        raw_image = cv2.resize(raw_image, (img_shape, img_shape))
        image = torch.FloatTensor(raw_image[..., ::-1].copy())
        image = image/255.0
        if da_model == "OPDA":
            image = _Normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #pdb.set_trace()
        #image -= model.mean
        #image /= model.std
        image = image.permute(2, 0, 1)
        return image, raw_image

    # Images
    def Load_txt(img_paths):
        assert isfile(img_paths),f"Image path {img_paths} doesn't exist"
        with open(img_paths,'rb') as fr:
            image_paths = [img_path.split()[0].decode("utf-8") for img_path in fr.readlines()]
        return image_paths

    def Filter_imgages(image_paths,req_images):
        img_files = list(map(lambda x:x.split('/')[-1].split('.')[0],image_paths))
        satisfied_images = []
        for item in req_images:
            try:
                [[idx]] = np.argwhere(np.isin(img_files,item))
            except:
                raise ValueError(f'{item} not found in the given paths')
            satisfied_images.append(image_paths[idx])
        return satisfied_images

    def Load_images(image_paths, req_images):
        if image_paths[0].endswith('.txt'):
            assert len(image_paths)==1 #make sure only one text file is given as input
            image_paths = Load_txt(image_paths[0])
            image_paths = Filter_imgages(image_paths, req_images)
            assert len(image_paths) == len(req_images)," All target images are not found"
        images = []
        raw_images = []
        print("Images:")
        for i, image_path in enumerate(image_paths):
            print("\t#{}: {}".format(i, image_path))
            image, raw_image = _preprocess(image_path, 224)
            images.append(image)
            raw_images.append(raw_image)
        images = torch.stack(images).to(device)
        return images,raw_images
    images, raw_images = Load_images(image_paths, target_images)
    print("Grad-CAM:")
    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    if not isdir(output_dir):
        os.makedirs(output_dir)
    for i in range(topk):

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "xception_v1", target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo4(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


if __name__ == "__main__":
    main()
