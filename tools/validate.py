import argparse

from groundingdino.config import get_cfg
from groundingdino.engine import Trainer


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Grounding DINO Validation')
    parser.add_argument('--config', type=str, default='groundingdino/config/GroundingDINO_SwinT_MultiModal.py')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    cfg = get_cfg()
    trainer = Trainer(cfg)
    trainer.load_checkpoint(args.checkpoint)
    val_loss = trainer.validate()
    print(f'Validation Loss: {val_loss:.4f}')


if __name__ == '__main__':
    main()