import argparse
import os
import sys
import importlib.util
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def import_groundingdino():
    groundingdino_path = os.path.join(project_root, 'groundingdino')
    
    init_path = os.path.join(groundingdino_path, '__init__.py')
    spec = importlib.util.spec_from_file_location('groundingdino', init_path)
    groundingdino = importlib.util.module_from_spec(spec)
    sys.modules['groundingdino'] = groundingdino
    spec.loader.exec_module(groundingdino)
    
    config_path = os.path.join(groundingdino_path, 'config', '__init__.py')
    spec_config = importlib.util.spec_from_file_location('groundingdino.config', config_path)
    config_module = importlib.util.module_from_spec(spec_config)
    sys.modules['groundingdino.config'] = config_module
    spec_config.loader.exec_module(config_module)
    
    engine_path = os.path.join(groundingdino_path, 'engine', '__init__.py')
    spec_engine = importlib.util.spec_from_file_location('groundingdino.engine', engine_path)
    engine_module = importlib.util.module_from_spec(spec_engine)
    sys.modules['groundingdino.engine'] = engine_module
    spec_engine.loader.exec_module(engine_module)
    
    return config_module.get_cfg, engine_module.Trainer


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Grounding DINO Training')
    parser.add_argument('--config', type=str, default='groundingdino/config/GroundingDINO_SwinT_MultiModal.py',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for inference-only loading')
    parser.add_argument('--resume', type=str, default=None, help='Path to latest_checkpoint.pth to resume training')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nproc_per_node', type=int, default=1, help='Number of processes per node')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master port')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    args = parser.parse_args()

    print(f"Project root: {project_root}")
    get_cfg, Trainer = import_groundingdino()
    
    cfg = get_cfg()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if os.path.exists(config_path):
        exec(open(config_path).read())
        cfg = get_cfg()
        print(f"Loaded config from: {config_path}")
    else:
        print(f"Warning: Config file not found: {config_path}")

    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir
    
    cfg.DISTRIBUTED.ENABLED = args.distributed
    cfg.LOCAL_RANK = args.local_rank
    cfg.MASTER_ADDR = args.master_addr
    cfg.MASTER_PORT = args.master_port

    trainer = Trainer(cfg)
    
    if args.resume is not None:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(project_root, resume_path)
        if os.path.exists(resume_path):
            print(f"Resuming training from: {resume_path}")
            trainer.train(resume_from=resume_path)
        else:
            print(f"Warning: Resume checkpoint not found: {resume_path}")
            print("Starting fresh training...")
            trainer.train()
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(project_root, checkpoint_path)
        trainer.load_checkpoint(checkpoint_path, load_training_state=False)
        trainer.train()
    else:
        latest_path = os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pth')
        if os.path.exists(latest_path):
            print(f"Found latest checkpoint, resuming from: {latest_path}")
            trainer.train(resume_from=latest_path)
        else:
            trainer.train()


if __name__ == '__main__':
    main()