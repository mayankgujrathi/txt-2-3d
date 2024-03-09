from setuptools import setup

setup(
    name="txt-2-3d",
    packages=[
        "mod",
        "mod.diff",
        "mod.prebuild",
        "mod.prebuild.generation",
        "mod.prebuild.nerf",
        "mod.prebuild.nerstf",
        "mod.prebuild.nn",
        "mod.prebuild.stf",
        "mod.prebuild.transmitter",
        "mod.meshlib",
        "mod.util",
    ],
    install_requires=[
        "scikit-image",
        "blobfile",
        "clip @ git+https://github.com/openai/CLIP.git",
        "numpy",
        "scipy",
        "fire",
        "matplotlib",
        "Pillow",
        "filelock",
        "tqdm",
        "requests",
        "torch",
        "pyyaml",
        "ipywidgets",
    ],
    author="Bhavana Kodali, Mayank Gujrathi",
)
