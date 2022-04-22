# CirBERTa - 搜狐情感分类baseline   (求star)

## Baseline实现

##### 基于pytorch，使用CirBERTa-Chinese预训练模型并基于prompt实现模型分类。

##### 仅包含NLP赛道内容，单模型无trick, 线上0.6996 

#####   



## 运行代码

#### 0.放置训练数据

##### 将Sohu2022_data放于baseline文件夹下



#### 1. 安装环境

```
pip install tqdm transformers pandas sklearn statistics

cd CirBERTa
pip install .
```



#### 2.训练（下载）模型

```
python download_model.py
python train_nlp.py
```



#### 2.测试模型

```
python test_nlp.py --model_name WENGSYX-CirBERTa-Chinese-Base/pytorch_model.bin
```

##### 注意：此处的model_name 应为训练完成的模型，请修改为paperlog文件夹下的pytorch_model.bin模型地址





## 引用:

(暂时先引用这个，论文正在撰写...)

```
@misc{CirBERTa,
  title={CirBERTa: Apply the Circular to the Pretraining Model},
  author={Yixuan Weng},
  howpublished={\url{https://github.com/WENGSYX/CirBERTa}},
  year={2022}
}
```
