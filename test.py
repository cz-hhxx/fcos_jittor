import os
import logging
import time
import numpy as np
from jittor.dataset import DataLoader
from sklearn.metrics import average_precision_score
from dataset import VOCDataset
from fcos import FCOS
from cfg import classes, scales, m, size
import jittor as jt
from utils import decode_heatmaps  # 导入解码函数

# 配置参数
WEIGHTS_PATH = "weights2/FCOS_epoch5_loss1.5398.pkl"
BATCH_SIZE = 2  # 测试批大小
LOG_DIR = "test_logs"  # 日志目录
DEVICE = "cuda"
NMS_THRESH = 0.5  # NMS的IoU阈值
CONF_THRESH = 0.1  # 置信度阈值


def init_logger():
    """初始化日志记录器"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"nms_test_{timestamp}.txt")

    logger = logging.getLogger("NMS_Test")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_gt_boxes(cls_maps_gt, loc_maps_gt, center_maps_gt, scales):
    """提取真实边界框（用于计算定位准确率）"""
    gt_boxes = decode_heatmaps(
        loc_maps_gt,
        center_maps_gt,
        cls_maps_gt,
        scales,
        thresh=0.0,  # 真实框不设置信度阈值
        use_nms=False  # 真实框无需NMS
    )
    return gt_boxes.numpy() if len(gt_boxes) > 0 else np.array([])


def get_pred_boxes(cls_maps_pred, loc_maps_pred, center_maps_pred, scales):
    """提取预测边界框（应用NMS）"""
    pred_boxes = decode_heatmaps(
        loc_maps_pred,
        center_maps_pred,
        cls_maps_pred,
        scales,
        thresh=CONF_THRESH,  # 过滤低置信度预测
        use_nms=True,
        nms_thresh=NMS_THRESH
    )
    return pred_boxes.numpy() if len(pred_boxes) > 0 else np.array([])

def test_with_nms():
    logger = init_logger()
    try:
        # 初始化Jittor
        jt.flags.use_cuda = 1

        # 1. 加载测试数据
        logger.info("加载测试集...")
        test_set = VOCDataset(root="data", train=False, size=size, scales=scales, m=m)
        test_set.set_attrs(batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set)
        total_samples = len(test_set)

        # 2. 加载模型
        logger.info("加载模型权重...")
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"未找到权重文件：{WEIGHTS_PATH}")

        model = FCOS(n_classes=len(classes))
        weight_dict = jt.load(WEIGHTS_PATH)  # Jittor加载权重
        model.load_state_dict(weight_dict["model"])
        model.eval()
        logger.info(f"模型已加载至{DEVICE}，开始推理...")

        # 3. 推理并收集评估数据
        all_metrics = {"tp": 0, "fp": 0, "fn": 0}  # 全局统计
        all_preds = []  # 用于AP计算的预测分数
        all_gts = []  # 用于AP计算的真实标签
        sample_idx = 0

        with jt.no_grad():
            for batch_idx, (imgs, loc_gt, center_gt, cls_gt, masks) in enumerate(test_loader):
                # 模型推理
                loc_pred, center_pred, cls_pred = model(imgs)

                # 逐样本处理
                for i in range(imgs.shape[0]):
                    if sample_idx >= total_samples:
                        break  # 防止超出样本总数

                    # 提取当前样本的预测框（应用NMS）
                    pred_boxes = get_pred_boxes(
                        [p[i] for p in cls_pred],
                        [l[i] for l in loc_pred],
                        [c[i] for c in center_pred],
                        scales
                    )

                    # 提取当前样本的真实框
                    gt_boxes = get_gt_boxes(
                        [c[i] for c in cls_gt],
                        [l[i] for l in loc_gt],
                        [ce[i] for ce in center_gt],
                        scales
                    )

                    # 收集AP计算所需数据（类别级）
                    gt_cls = np.zeros(len(classes))
                    for gt in gt_boxes:
                        gt_cls[int(gt[0])] = 1
                    all_gts.append(gt_cls)

                    pred_cls = np.zeros(len(classes))
                    for pred in pred_boxes:
                        cls_idx = int(pred[0])
                        if pred[1] > pred_cls[cls_idx]:
                            pred_cls[cls_idx] = pred[1]
                    all_preds.append(pred_cls)

                    sample_idx += 1

        # 4. 计算全局评估指标
        logger.info("\n===== 所有样本测试完成，开始计算指标 =====")

        # 4.2 计算每类AP和mAP
        logger.info("\n===== 类别级AP指标 =====")
        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)
        ap_list = []

        for i in range(len(classes)):
            cls_name = classes[i]
            y_pred = all_preds[:, i]
            y_gt = all_gts[:, i]

            if np.sum(y_gt) == 0:
                logger.info(f"{cls_name}: 无正样本，AP=0.0")
                ap_list.append(0.0)
                continue

            ap = average_precision_score(y_gt, y_pred)
            ap_list.append(ap)
            logger.info(f"{cls_name}: AP={ap:.4f}")

        mAP = np.mean(ap_list)
        logger.info(f"\nmAP50: {mAP:.4f}")

    except Exception as e:
        logger.error(f"测试失败：{str(e)}", exc_info=True)
    finally:
        logger.info("所有测试流程结束")


if __name__ == "__main__":
    test_with_nms()
