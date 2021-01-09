import torch
import argparse
import os
import sys
sys.path.append(os.path.abspath('../../'))
from arch.layout2im.generator import Generator
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from data.coco_custom_mask import get_dataloader as get_dataloader_coco
from data.custom_dataloader import get_dataloader as get_dataloader_custom
from utils.data import imagenet_deprocess_batch
from imageio import imwrite
from pathlib import Path
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw
import numpy as np
from data.preprocess import object_names

def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

def main(config):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result_save_dir = config.results_dir
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)

    if config.dataset == 'vg':
        train_data_loader, val_data_loader = get_dataloader_vg(batch_size=config.batch_size, VG_DIR=config.vg_dir)
    elif config.dataset == 'coco':
        train_data_loader, val_data_loader = get_dataloader_coco(batch_size=config.batch_size, COCO_DIR=config.coco_dir)
    elif config.dataset == 'custom':
        train_data_loader, val_data_loader= get_dataloader_custom(batch_size=config.batch_size, CUSTOM_DIR='.' )
    vocab_num = train_data_loader.dataset.num_objects

    assert config.clstm_layers > 0
    netG = Generator(num_embeddings=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim, clstm_layers=config.clstm_layers).to(device)

    print('load model from: {}'.format(config.saved_model))
    netG.load_state_dict(torch.load(config.saved_model))

    data_loader = val_data_loader
    data_iter = iter(data_loader)
    with torch.no_grad():
        netG.eval()
        for i, batch in enumerate(data_iter):
            print('batch {}'.format(i))
            imgs, objs, boxes, masks, classes, obj_to_cls, obj_to_img = batch
            z = torch.randn(objs.size(0), config.z_dim)
            imgs, objs, boxes, masks, classes, obj_to_cls, obj_to_img, z = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), classes.to(device), obj_to_cls.to(device), obj_to_img, z.to(device)

            # Generate fake image
            output = netG(imgs, objs, boxes, masks, classes, obj_to_img, z)
            crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec = output

            img_rand = imagenet_deprocess_batch(img_rand)
            class_cpu = classes.cpu()

            #img_rand = img_rand.cpu()
            # Save the generated images
            print(class_cpu.shape)
            for j in range(img_rand.shape[0]):
           
                class_val = np.where(class_cpu[j]==1)
                #print(class_cpu.shape)
                 #print(class_val[0][0])
                cls_name = object_names[class_val[0][0]]
                print(cls_name)  
                img_np = img_rand[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, '{}_img{:06d}.png'.format(str(cls_name),i*config.batch_size+j))
                #box_path = os.path.join(result_save_dir, 'box{:06d}.png'.format(i*config.batch_size+j))
                
                imwrite(img_path, img_np)
                #img = Image.fromarray(img_np)
                #draw = ImageDraw.Draw(img)
                #for k in range(boxes.shape[0]):
                # x1, y1, x2, y2 = boxes[k] * 128.0
                # print('img shape {}  {} {} {} {}'.format(img_np.shape, x1, y1, x2, y2))
                # rect = get_rect(x=x1, y=y1, width=x2-x1+1, height=y2-y1+1, angle=0.0)
                # draw.polygon([tuple(p) for p in rect], fill=None)
                #img_np = np.asarray(img)
                
                #imwrite(box_path, img_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datasets configuration
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--vg_dir', type=str, default='datasets/vg')
    parser.add_argument('--coco_dir', type=str, default='datasets/coco')

    # Model configuration
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--object_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--resi_num', type=int, default=6)
    parser.add_argument('--clstm_layers', type=int, default=3)

    # Model setting
    parser.add_argument('--saved_model', type=str, default='checkpoints/pretrained/netG_coco.pkl')

    config = parser.parse_args()
    config.results_dir = 'checkpoints/boxvae_layout2im_white_gen_{}'.format(config.dataset)

    print(config)

    main(config)
