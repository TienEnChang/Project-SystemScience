# Overview

This project is based on 'the DeepMind's AlphaGo Zero project'. 

To see the original GitHub implementation 'Alpha Zero General', please visit:\
[https://github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)\
\
To install related docker images: (vscode ver_1.18.0)\
[https://hub.docker.com/r/continuumio/miniconda3](https://hub.docker.com/r/continuumio/miniconda3)\
[https://hub.docker.com/r/jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook)\
[https://phoenixnap.com/kb/how-to-commit-changes-to-docker-image](https://phoenixnap.com/kb/how-to-commit-changes-to-docker-image)

```
docker run -it -d -v /Users/tim/Documents/Programming:/root continuumio/miniconda3:usable
```
```
docker run -it -d -v $(pwd):/root continuumio/miniconda3
docker pull --platform=linux/amd64 continuumio/miniconda3
```
