# Neural Question Generation
This is not official implementation from the paper [Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks](https://www.aclweb.org/anthology/D18-1424)
 I implemented in Pytorch to reproduce similar result as the paper

## Dependencies
This code is written in Python. Dependencies include
* python >= 3.6
* pytorch >= 1.0
* nltk
* tqdm
* [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)

## Download data and Preprocess
```bash
mkdir squad
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ./data/glove.840B.300d.zip 
unzip ./data/glove.840B.300d.zip 
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O ./squad/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./squad/dev-v1.1.json
cd data
python process_data.py
```
## Configuration
You might need to change configuration in config.py. <br />
If you want to train, change train = True  and set gpu device in config.py


## Evaluation from this [repository](https://github.com/xinyadu/nqg)
```bash
cd qgevalcap
python2 eval.py --out_file prediction_file --src_file src_file --tgt_file target_file
``` 
## Results 
|  <center>BLEU_1</center> |  <center>BLEU_2</center> |  <center>BLEU_3</center> | <center>BLEU_4</center> |
|:--------|:--------:|--------:|--------:|
|<center>44.575 </center> | <center>29.34 </center> |<center> 21.74</center>| <center>16.28</center>|