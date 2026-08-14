"""快速验证 loss 修复：构造模拟数据，检查 loss 值是否在合理范围内。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from groundingdino.models.utils import box_cxcywh_to_xyxy, generalized_box_iou

print("=" * 60)
print("验证 1: box_cxcywh_to_xyxy 转换正确性")
print("=" * 60)

# 模拟 cxcywh 归一化框
boxes_cxcywh = torch.tensor([
    [0.5, 0.5, 0.4, 0.4],   # 中心在图像中央，40%x40% 大小
    [0.3, 0.3, 0.2, 0.2],   # 左上角小框
    [0.7, 0.7, 0.3, 0.3],   # 右下角中框
])
boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
print(f"  cxcywh: {boxes_cxcywh.tolist()}")
print(f"  xyxy:   {boxes_xyxy.tolist()}")
# 验证: [0.5, 0.5, 0.4, 0.4] → [0.3, 0.3, 0.7, 0.7]
assert abs(boxes_xyxy[0, 0].item() - 0.3) < 1e-6, "转换错误"
assert abs(boxes_xyxy[0, 2].item() - 0.7) < 1e-6, "转换错误"
print("  ✅ 转换正确\n")

print("=" * 60)
print("验证 2: GIoU 在 xyxy 格式下的值域")
print("=" * 60)

# 相同的框 → GIoU 应该 = 1, loss_giou = 1 - 1 = 0
giou_same = generalized_box_iou(boxes_xyxy, boxes_xyxy)
diag_giou_same = torch.diag(giou_same)
print(f"  相同框 GIoU 对角线: {diag_giou_same.tolist()}")
print(f"  loss_giou (1-GIoU): {(1 - diag_giou_same).tolist()}")
assert all(abs(v - 1.0) < 1e-4 for v in diag_giou_same), "相同框 GIoU 应为 1"
print("  ✅ 相同框 GIoU=1, loss_giou=0\n")

# 完全不重叠的框 → GIoU 应 < 0, loss_giou > 1
box_a = torch.tensor([[0.1, 0.1, 0.2, 0.2]])  # 左上角小框
box_b = torch.tensor([[0.8, 0.8, 0.9, 0.9]])  # 右下角小框
giou_no_overlap = generalized_box_iou(box_a, box_b)
print(f"  不重叠框 GIoU: {giou_no_overlap.item():.4f}")
print(f"  loss_giou (1-GIoU): {1 - giou_no_overlap.item():.4f}")
assert giou_no_overlap.item() < 0, "不重叠框 GIoU 应为负"
assert (1 - giou_no_overlap.item()) < 3, "loss_giou 应在合理范围 [0, 3]"
print("  ✅ 不重叠框 GIoU<0, loss_giou 在合理范围\n")

# 部分重叠的框 → GIoU 应在 [-1, 1], loss_giou 在 [0, 2]
box_c = torch.tensor([[0.2, 0.2, 0.6, 0.6]])
box_d = torch.tensor([[0.4, 0.4, 0.8, 0.8]])
giou_partial = generalized_box_iou(box_c, box_d)
print(f"  部分重叠框 GIoU: {giou_partial.item():.4f}")
print(f"  loss_giou (1-GIoU): {1 - giou_partial.item():.4f}")
assert -1 <= giou_partial.item() <= 1, "GIoU 应在 [-1, 1]"
assert 0 <= (1 - giou_partial.item()) <= 2, "loss_giou 应在 [0, 2]"
print("  ✅ 部分重叠框 GIoU 在 [-1,1], loss_giou 在 [0,2] 合理\n")

print("=" * 60)
print("验证 3: 旧 bug 复现（cxcywh 直接传入 GIoU 的错误值）")
print("=" * 60)
# 旧 bug: 直接传 cxcywh 给 generalized_box_iou
giou_buggy = generalized_box_iou(boxes_cxcywh, boxes_cxcywh)
diag_buggy = torch.diag(giou_buggy)
print(f"  旧 bug (cxcywh 直接传入) GIoU: {diag_buggy.tolist()}")
print(f"  旧 bug loss_giou: {(1 - diag_buggy).tolist()}")
if any(abs(v) > 10 for v in diag_buggy):
    print("  ❌ 旧 bug 产生异常值（这正是之前 loss=-1M 的原因）")
else:
    print("  ⚠️ 旧 bug 此组数据未产生极端值（但仍然不正确）")
print()

print("=" * 60)
print("✅ 所有验证通过！GIoU 修复正确，loss 值将在合理范围内。")
print("=" * 60)
