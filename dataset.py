import math
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from jittor import transform
import cv2
import xml.etree.ElementTree as ET

from cfg import classes


# 替换VOC数据集加载
class VOCDetection:
    def __init__(self, root, image_set='train'):
        # 数据集根目录
        self.root = root
        # 图像集类型
        self.image_set = image_set
        # 图像文件所在目录
        self.images_path = f"{root}/VOCdevkit/VOC2007/JPEGImages"
        # 标注文件所在目录
        self.annotations_path = f"{root}/VOCdevkit/VOC2007/Annotations"

        # 加载图像列表
        self.ids = []
        # 打开图像集列表
        with open(f"{root}/VOCdevkit/VOC2007/ImageSets/Main/{image_set}.txt", 'r') as f:
            # 找到所有包含该类别的图片的ID
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    # 定义数据集长度
    def __len__(self):
        return len(self.ids)

    # 根据索引idx获取图像ID
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # 加载图像
        img = cv2.imread(f"{self.images_path}/{img_id}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 加载标注
        tree = ET.parse(f"{self.annotations_path}/{img_id}.xml")
        root = tree.getroot()

        # 使用嵌套字典模仿 VOC XML的格式
        target = {'annotation': {'object': []}}
        # 遍历XML中所有的object节点
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            target['annotation']['object'].append({
                'name': cls_name,
                'bndbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            })

        return img, target


class VOCDataset(Dataset):
    def __init__(self,
                 root,
                 download=True,  # 注意jittor并未提供直接可下载的数据
                 train=True,
                 size=(800, 1024),
                 scales=(8, 16, 32, 64, 128),  # 多尺度特征层的缩放因子大小
                 multi_scale=True,  # 是否按目标大小分配到不同的特征层
                 m=(0, 32, 64, 128, 256, np.inf),  # 多尺度分配的距离阈值
                 center_sampling=False,  # 是否中心采样
                 radius=2):  # 中心采样半径
        super().__init__()

        if train:
            image_set = 'train'
        else:
            image_set = 'val'

        self.base = VOCDetection(root, image_set=image_set)
        self.scales = scales
        self.m = m
        self.radius = radius

        self.multi_scale = multi_scale
        self.center_sampling = center_sampling

        self.size = size

        # 使用Jittor内置的变换组合
        self.trans = transform.Compose([
            transform.Resize(size),  # 图像缩放
            transform.ToTensor(),  # numpy变为tensor
            transform.ImageNormalize(mean=[.485, .456, .406], std=[.229, .224, .225]),  # 替换ToTensor和Normalize
        ])

        # 反归一化变换
        # self.untrans = transform.Compose([
        #     transform.ImageNormalize(mean=[-.485 / .229, -.456 / .224, -.406 / .225],
        #                              std=[1 / .229, 1 / .224, 1 / .225])  # 替换自定义UnNormalize
        # ])

        # Jittor数据集需要设置batch_size和shuffle（在train那里讲一下）
        self.batch_size = 2
        self.shuffle = train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # 从基础数据集加载原始图像和标注
        img, target = self.base[idx]
        # 解析字典，提取目标的类别和边界框
        boxes = self._parse_target_dict(target)
        # 对图像进行预处理，
        img, boxes = self._resize_and_sort(img, boxes)

        # 初始化列表，存储每个尺度特征层的目标（位置、中心度、类别、掩码）
        loc_maps, center_maps, cls_maps, masks = [], [], [], []
        # 遍历每一个尺度，生成对应的特征层目标
        for idx, scale in enumerate(self.scales):
            boxes_ = np.array(boxes)
            boxes_[:, 1:] = boxes_[:, 1:] / scale  # 仅对坐标进行不同尺度的缩放
            # 对该张图片的当前尺度生成热图：（位置回归目标、中心度目标、类别目标、正样本掩码）
            # 输入的是当前尺度的坐标值、缩放因子、多尺度分配的距离范围、中心采样半径
            loc_map, center_map, cls_map, mask = self._gen_heatmap(
                boxes_, scale, self.m[idx], self.m[idx + 1], self.radius)
            loc_maps.append(loc_map)
            center_maps.append(center_map)
            cls_maps.append(cls_map)
            masks.append(mask)

        return img, loc_maps, center_maps, cls_maps, masks

    def _parse_target_dict(self, target):
        boxes = []  # 存储边界框：(类别索引，xmin，ymin，xmax，ymax )
        for obj in target['annotation']['object']:
            cls = obj['name']
            cls = classes.index(cls)
            box = obj['bndbox']
            xmin, ymin, xmax, ymax = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
            box = (cls, xmin, ymin, xmax, ymax)  # 列表中的每个元素包含5个信息，类别名称，以及左上和右下角的坐标
            boxes.append(box)
        return boxes

    # 对图像进行预处理，调整边界框坐标，并按面积排序
    def _resize_and_sort(self, img, boxes):
        h_origin, w_origin = img.shape[:2]
        # 应用变换后，显式转换为 Jittor 张量（若变换未自动转换）
        img = self.trans(img)
        if isinstance(img, np.ndarray):
            img = jt.array(img)  # 确保是 Jittor 张量
        h, w = self.size
        for ind, box in enumerate(boxes):
            boxes[ind] = (box[0],
                          box[1] * w / w_origin,
                          box[2] * h / h_origin,
                          box[3] * w / w_origin,
                          box[4] * h / h_origin)
        # 按边界框面积从小到大进行排序（小目标优先分配到高分辨率特征层）
        boxes.sort(key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))
        return img, boxes

    def _gen_heatmap(self, boxes, scale, m_lb, m_ub, radius):
        h, w = np.ceil(np.array(self.size) / scale).astype(int)

        loc_map = np.zeros((4, h, w)).astype(float)
        center_map = np.zeros((h, w)).astype(float)
        cls_map = np.zeros((len(classes), h, w))

        all_mask = np.zeros((h, w))

        x, y = np.meshgrid(range(w), range(h))

        for box in boxes:
            cls, xmin, ymin, xmax, ymax = box
            cls = int(cls)

            tmp_mask = np.zeros((h, w)).astype(int)
            tmp_mask[int(math.ceil(ymin)):int(math.floor(ymax)) + 1,
            int(math.ceil(xmin)):int(math.floor(xmax)) + 1] = 1

            l = x - xmin
            t = y - ymin
            r = xmax - x
            b = ymax - y
            l[l < 0] = 0
            t[t < 0] = 0
            r[r < 0] = 0
            b[b < 0] = 0
            l *= tmp_mask
            t *= tmp_mask
            r *= tmp_mask
            b *= tmp_mask

            if self.multi_scale:
                dist = np.max(np.stack([l, t, r, b]), 0) * scale
                tmp_mask = np.where((m_lb <= dist) & (dist <= m_ub), 1, 0)

            # if self.center_sampling:
            #     center_mask = np.zeros((h, w)).astype(int)
            #     xc, yc = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
            #     center_mask[yc - radius:yc + radius + 1, xc - radius:xc + radius + 1] = 1
            #     tmp_mask *= center_mask

            tmp_mask = np.where(tmp_mask > all_mask, 1, 0)
            all_mask += tmp_mask

            cls_map[cls, :, :] += tmp_mask.copy() * (cls + 1)

            loc_map[0, :, :] += l * tmp_mask
            loc_map[1, :, :] += t * tmp_mask
            loc_map[2, :, :] += r * tmp_mask
            loc_map[3, :, :] += b * tmp_mask

            min_lr = np.where(l > r, r, l)
            inv_max_lr = np.where(l > r, 1 / (l + 1e-8), 1 / (r + 1e-8))
            min_tb = np.where(t > b, b, t)
            inv_max_tb = np.where(t > b, 1 / (t + 1e-8), 1 / (b + 1e-8))
            center_map += np.sqrt(min_lr * inv_max_lr *
                                  min_tb * inv_max_tb) * tmp_mask

        loc_map = jt.array(loc_map)
        center_map = jt.array(center_map).clamp(0., 1.)
        cls_map = jt.array(cls_map).sum(0)
        all_mask = jt.array(all_mask).clamp(0, 1).astype(jt.bool)

        return loc_map, center_map, cls_map, all_mask
