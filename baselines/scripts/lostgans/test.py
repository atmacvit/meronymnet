import argparse
from collections import OrderedDict
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cocostuff_loader import *
from data.vg import *
from data.custom_loader import *
from data.preprocess import *
from model.resnet_generator_v2_condition import *
from utils.util import *
from PIL import Image, ImageDraw

def get_dataloader(dataset = 'coco', img_size=128, classes=None):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/images/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json=None,#'./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=False, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        with open("./datasets/vg/vocab.json", "r") as read_file:
            vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab=vocab,
                                      h5_path='./datasets/vg/val.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(128, 128), left_right_flip=False, max_objects=30)
    else:
        dataset = CustomDataset(image_dir='./datasets/coco/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json=None,#'./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=False, image_size=(img_size, img_size), left_right_flip=False, data_mode='test', classes=classes.split(','))

    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=True, shuffle=False, num_workers=1)
    return dataloader

def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

def main(args):
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31
    num_classes = 25 if args.dataset == 'custom' else num_classes
    num_o = 25 if args.dataset == 'custom' else num_o
    dataloader = get_dataloader(args.dataset, classes=args.classes)

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()

    if not os.path.isfile(args.model_path):
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres=2.0
    for idx, data in enumerate(dataloader):
        real_images, label, bbox, cls = data
        print("Shape " + str(bbox.shape))
        print("BBox " + str(bbox))
        real_images, label, cls = real_images.cuda(), label.long().unsqueeze(-1).cuda(), cls.cuda()
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
        fake_images, stage_bbox, seman_bbox, var, alpha = netG.forward(z_obj, bbox.cuda(), cls, z_im, label.squeeze(dim=-1))
        print("Stage bbox Shape {}".format(stage_bbox.shape))
        print("Seman bbox Shape {}".format(seman_bbox.shape))
        print("Var Shape {}".format(var.shape))
        masks = seman_bbox.cpu().detach().numpy()
        #bbx = bbox.cpu().detach().numpy()
        #print(bbox.shape)
        class_cpu = cls.cpu()
        class_val = np.where(class_cpu[0]==1)
        #print(class_cpu.shape)
        #print(class_val[0][0])
        img_np = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)
        cls_name = object_names[class_val[0][0]]
        stage_bbox = stage_bbox.cpu().detach().numpy()
        seman_bbox = seman_bbox.cpu().detach().numpy()
        var = var.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        #if cls_name=='cat' and idx==920:
        #  #print(label)
          #print(bbox)
        #  for i in range(25):
           #misc.imsave("masks_lostgan/seman_bbox/{}_{}.png".format(idx, i),seman_bbox[0][i])
           #misc.imsave("masks_lostgan/stage_bbox/{}_{}.png".format(idx, i),stage_bbox[0][i])
            #misc.imsave("masks_lostgan/Var/{}_{}.png".format(idx, i),var[0][i])
            #misc.imsave("masks_lostgan/alpha4/{}_{}.png".format(idx, i),alpha[0][i])

    #      return

            
        
        if idx>=0:
         print("Entered")
         canvas = np.zeros((64, 64), dtype=np.float32)
         canvas_sum = np.zeros((64, 64), dtype=np.float32) 
         for i in range(64):
          for j in range(64):
           canvas[i][j] = np.max(masks[0, :, i, j])
         #for i in range(25):
         # masks[0][i] = bounder(masks[0][i])
         #bbx[idx] = bbx[idx] * 550.0
         #bbx[idx][2] += bbx[idx][0]-1
         #bbx[idx][3] += bbx[idx][1]-1
         #canvas = make_mask(bbx[idx], masks[idx], cls_name)
         # if label[0][i][0]!=0:
         #  print(i)
         #  print("Seman Min {}  Max {}".format(np.min(masks[0][i]), np.max(masks[0][i])))
         #  coord = np.where(masks[0][i]!=0)
         #  for cord in list(zip(coord[0],coord[1])):
         #    val= masks[0][i][cord]
         #    canvas_sum[coord] += val
         #    if val>canvas[cord]:
         #     canvas[cord]=val
         canvas_bounder = bounder(canvas)
         canvas_wobounder = canvas
         misc.imsave("mask_{}_sample_{}.png".format(cls_name, idx),canvas_bounder)
         #misc.imsave("{}_samplewobounder_{}.png".format(cls_name, idx),canvas_wobounder)
         #misc.imsave("{}_samplewoboundersum_{}.png".format(cls_name, idx),canvas_sum)
         #return 
        img = Image.fromarray((img_np*255.0).astype('uint8'))
        draw = ImageDraw.Draw(img)
        for k in range(bbox[0].shape[0]):
         x1, y1, x2, y2 = bbox[0][k] * 128.0
         print('img shape {}  {} {} {} {}'.format(img_np.shape, x1, y1, x1 + x2 - 1, y1 + y2 - 1))
         rect = get_rect(x=x1, y=y1, width=x2, height=y2, angle=0.0)
         draw.polygon([tuple(p) for p in rect], fill=None)
         img_np_bbx = np.asarray(img)

        misc.imsave("{save_path}/{class_name}_sample_{idx}.jpg".format(save_path=args.sample_path, class_name=cls_name, idx=idx),
                    img_np)
        #misc.imsave("{save_path}/{class_name}_sample_bbx_{idx}.jpg".format(save_path=args.sample_path, class_name=cls_name, idx=idx),
                    #img_np_bbx)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    parser.add_argument('--classes', type=str, default='dog')
    args = parser.parse_args()
    main(args)
