from distutils.core import setup

setup(
    name="infMachine",
    packages=["src"],
    license="",
    python_requires="~=3.9",
    install_requires=[
        "torchvision==0.17.0",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "scikit-learn==1.1.1",
        "hydra-core==1.3.2",
        "wandb==0.16.3",
        "matplotlib==3.8.0",
        "attrdict==2.0.1",
        "hydra-submitit-launcher==1.2.0",
        "seaborn==0.13.2",
        "sbi==0.22.0"
    ],
)
