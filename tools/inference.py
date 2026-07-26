import argparse
import cv2
import matplotlib.pyplot as plt

from groundingdino.config import get_cfg
from groundingdino.inference import GroundingPredictor


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Grounding DINO Inference')
    parser.add_argument('--config', type=str, default='groundingdino/config/GroundingDINO_SwinT_MultiModal.py')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--rgb', type=str, default=None)
    parser.add_argument('--ir', type=str, default=None)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    cfg = get_cfg()
    predictor = GroundingPredictor(cfg, args.checkpoint)

    results = predictor.predict(
        rgb_path=args.rgb,
        ir_path=args.ir,
        depth_path=args.depth,
        text_prompt=args.text
    )

    if args.rgb is not None:
        vis_img = predictor.visualize(args.rgb, results, args.threshold)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.show()

    print('Inference completed!')


if __name__ == '__main__':
    main()