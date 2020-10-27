from setuptools import setup, find_packages
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lipsync',
    version='0.0.1',
    description='Lip Synchronization (Wav2Lip).',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/mowshon/flarepy',
    keywords='lipsync, lip, wav2lip, lip synchronization',
    author='Rudrabha Mukhopadhyay, Mowshon',
    author_email='mowshon@yandex.ru',
    license='MIT',
    packages=[
        'lipsync',
    ],
    package_data={
          'lipsync': [
              '*',
              'Wav2Lip/*'
          ],
      },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'librosa==0.7.0',
        'numpy',
        'opencv-contrib-python==4.2.0.34',
        'opencv-python==4.1.0.25',
        'tensorflow==1.13.1',
        'torch==1.1.0',
        'torchvision==0.3.0',
        'tqdm==4.45.0',
        'numba==0.48'
    ],
    zip_safe=False
)