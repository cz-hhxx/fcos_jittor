import jittor as jt
from jittor import optim
from jittor import lr_scheduler
from jittor.dataset import DataLoader
from tqdm import tqdm
from visdom import Visdom
import time
import logging
import os

from dataset import VOCDataset
from fcos import FCOS
from loss import Loss
from cfg import scales, m, size

# ------------------- 日志配置 -------------------
# 设置日志文件夹
log_dir = 'logging'
os.makedirs(log_dir, exist_ok=True)

# 创建唯一的日志文件名（包含时间戳）
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'train_log_{timestamp}.txt')

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 每次运行覆盖旧日志
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ------------------------------------------------

# 初始化Jittor
jt.flags.use_cuda = jt.has_cuda

batch_size = 1
show_every = 20
lr = 1e-4
epochs = 10
# history_weights = 'FCOS_epoch3_loss1.6628.pth'

device = "cuda" if jt.has_cuda else "cpu"
logger.info(f'Use device: {device}')

train_set = VOCDataset(root='data', train=True, size=size, scales=scales, m=m)
train_set.set_attrs(batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
logger.info(f'Train loader length: {len(train_loader)}')

# 挑选测试图片
test_set = VOCDataset(root='data', train=False, size=size, scales=scales, m=m)
img_test, loc_maps_test, center_maps_test, cls_maps_test, _ = test_set[48]
img_test = img_test.unsqueeze(0)

model = FCOS(n_classes=20)
criterion = Loss()

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12], gamma=0.1)

history_ep = 0


viz = Visdom()
viz.line([[0, 0, 0, 0]], [0], win='train_loss', opts=dict(
    title='Train Loss', legend=['total', 'loc', 'center', 'cls']))

# 用于记录整个epoch的平均损失
final_loss = 0.0

for ep in range(history_ep + 1, epochs + 1):
    logger.info(f'--- Starting Epoch {ep} ---')
    model.train()
    loc_loss_total, center_loss_total, cls_loss_total, count = 0., 0., 0., 0

    pbar = tqdm(train_loader, desc=f'Epoch {ep}', leave=False)
    for index, (imgs, loc_maps, center_maps, cls_maps, masks) in enumerate(pbar):
        # 前向传播
        loc_maps_pred, center_maps_pred, cls_maps_pred = model(imgs)

        # 计算损失
        loc_loss, center_loss, cls_loss = criterion(loc_maps_pred, loc_maps,
                                                    center_maps_pred, center_maps,
                                                    cls_maps_pred, cls_maps,
                                                    masks)

        loss = loc_loss + center_loss + cls_loss

        # 反向传播（Jittor 版本：一步完成）
        optimizer.step(loss)

        loc_loss_total += loc_loss.item()
        center_loss_total += center_loss.item()
        cls_loss_total += cls_loss.item()

        count += 1
        if count == show_every or index == len(train_loader) - 1:
            total_loss = loc_loss_total + center_loss_total + cls_loss_total
            avg_total = total_loss / count
            avg_loc = loc_loss_total / count
            avg_center = center_loss_total / count
            avg_cls = cls_loss_total / count

            # 更新 Visdom
            viz.line([[avg_total, avg_loc, avg_center, avg_cls]],
                     [(ep - 1) * len(train_loader) + index], 'train_loss', update='append')

            # 记录日志
            logger.info(f'Epoch {ep}, Step {index + 1}, Avg Loss: {avg_total:.4f} '
                        f'(Loc: {avg_loc:.4f}, Center: {avg_center:.4f}, Cls: {avg_cls:.4f})')

            if index == len(train_loader) - 1:
                final_loss = avg_total

            # 重置
            loc_loss_total, center_loss_total, cls_loss_total, count = 0., 0., 0., 0

            # 可视化测试集结果（降低频率）
            if index % (show_every * 10) == 0 or index == len(train_loader) - 1:
                layer = 2


    # 更新学习率
    lr_scheduler.step()

    # 保存模型
    save_path = f'weights/FCOS_epoch{ep}_loss{final_loss:.4f}.pkl'
    jt.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        # 'lr_scheduler': lr_scheduler.state_dict(),  # 可选保存
    }, save_path)
    logger.info(f'Epoch {ep} completed. Final Avg Loss: {final_loss:.4f}. Model saved to {save_path}')

logger.info('Training finished.')