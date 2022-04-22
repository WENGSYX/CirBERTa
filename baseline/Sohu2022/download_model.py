import os
import json
import requests
from uuid import uuid4
from tqdm import tqdm


SESSIONID = uuid4().hex

CONFIG_FILE = "config.json"
MODEL_FILE = "pytorch_model.bin"
BASE_URL = "https://huggingface.co/{}/resolve/main/{}"

headers = {'user-agent': 'transformers/4.8.2; python/3.8.5;  \
			session_id/{}; torch/1.9.0; tensorflow/2.5.0; \
			file_type/model; framework/pytorch; from_auto_class/False'.format(SESSIONID)}

model_id = "WENGSYX/CirBERTa-Chinese-Base"

# 创建模型对应的文件夹

model_dir = model_id.replace("/", "-")

if not os.path.exists(model_dir):
	os.mkdir(model_dir)


r = requests.get(BASE_URL.format(model_id, CONFIG_FILE), headers=headers)
r.encoding = "utf-8"
with open(os.path.join(model_dir, CONFIG_FILE), "w", encoding="utf-8") as f:
	json.dump(r.json(), f, indent="\t")
	print("{}配置文件下载完毕!".format(model_id))


# 模型文件需要分两步进行

# Step1 获取模型下载的真实地址
r = requests.head(BASE_URL.format(model_id, MODEL_FILE), headers=headers)
r.raise_for_status()
if 300 <= r.status_code <= 399:
	url_to_download = r.headers["Location"]

# Step2 请求真实地址下载模型
r = requests.get(url_to_download, stream=True, proxies=None, headers=None)
r.raise_for_status()

# 这里的进度条是可选项，直接使用了transformers包中的代码
content_length = r.headers.get("Content-Length")
total = int(content_length) if content_length is not None else None
progress = tqdm(
	unit="B",
	unit_scale=True,
	total=total,
	initial=0,
	desc="Downloading Model",
)

with open(os.path.join(model_dir, MODEL_FILE), "wb") as temp_file:
	for chunk in r.iter_content(chunk_size=1024):
		if chunk:  # filter out keep-alive new chunks
			progress.update(len(chunk))
			temp_file.write(chunk)

progress.close()

print("{}模型文件下载完毕!".format(model_id))