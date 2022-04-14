# CirBERTa
Apply the Circular to the Pretraining Model

## CBLUE榜单第一名模型
### 使用通用语料（WuDao 200G) 进行无监督预训练


| 预训练模型            | 学习率 | batchsize | 设备   | 语料库 | 时间 | 优化器 |
| --------------------- | ------ | --------- | ------ | ------ | ---- | ------ |
| CirBERTa-Chinese-Base | 1e-5   | 256      | 10*3090+3*A100 | 200G   | 2月 | AdamW  |

在多项任务上，CirBERTa-Base模型超过MacBERT-Large/RoBERTa-Large

<center><img src="img/1.png" alt="img" style="zoom:50%;" /></center>

### 加载与使用

依托于huggingface-transformers

```
from transformers import AutoTokenizer,AutoModel

tokenizer = AutoTokenizer.from_pretrained("WENGSYX/CirBERTa-Chinese-Base")
model = AutoModel.from_pretrained("WENGSYX/CirBERTa-Chinese-Basee")