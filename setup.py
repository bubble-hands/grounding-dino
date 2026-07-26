from setuptools import setup, find_packages

setup(
    name='groundingdino',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.30.0',
        'timm>=0.9.0',
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'Pillow>=10.0.0',
        'scipy>=1.10.0',
        'pyyaml>=6.0.0',
        'matplotlib>=3.7.0',
        'wandb>=0.15.0',
        'termcolor>=2.3.0',
        'pycocotools>=2.0.6',
    ],
)