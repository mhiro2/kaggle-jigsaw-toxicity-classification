# kaggle-jigsaw-toxicity-classification

Part of 14th place solution to [Jigsaw Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). (slightly modified by refactoring)
- Public LB: 12th (0.94655)
- Private LB: 14th (0.94628)

## Prerequisite
- Pull PyTorch image from [NVIDIA GPU CLOUD (NGC)](https://ngc.nvidia.com/)
  ```
  docker login nvcr.io
  docker image pull nvcr.io/nvidia/pytorch:19.04-py3
  ```

## Usage

### BERT-Base
```
# NOTE: Apex ver. 0.1 is already installed in this image
docker container run -it --name=bert --runtime=nvidia --ipc=host -v $PWD:/workspace/jigsaw nvcr.io/nvidia/pytorch:19.04-py3
```
```
cd /workspace/jigsaw/src

pip install pytorch_pretrained_bert==0.6.2
pip install fastprogress

# train BERT model
python train_bert_base_full.py
```

### GPT-2
```
docker container run -it --name=gpt2 --runtime=nvidia --ipc=host -v $PWD:/workspace/jigsaw nvcr.io/nvidia/pytorch:19.04-py3
```
```
cd /workspace/jigsaw/src

pip install git+https://github.com/pronkinnikita/pytorch-pretrained-BERT
pip install fastprogress

# train GPT-2 model
python train_gpt2_full.py
```

## Author

[Masaaki Hirotsu](<mailto:hirotsu.masaaki@gmail.com>) / Kaggle: [@mhiro2](https://www.kaggle.com/mhiro2)

