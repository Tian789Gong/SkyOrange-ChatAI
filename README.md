![](img/logo-long-chatchat-trans-v2.png)

🌍 [READ THIS IN ENGLISH](README_en.md)
🌍 [日本語で読む](README_ja.md)

📃 **SkyOrange-ChatAI** 

基于 ChatGLM 等大语言模型与 Langchain 等应用框架实现，开源、可离线部署的检索增强生成(RAG)大模型知识库项目。


---

## 目录

* [介绍](README.md#介绍)
* [解决的痛点](README.md#解决的痛点)
* [快速上手](README.md#快速上手)
    * [1. 环境配置](README.md#1-环境配置)
    * [2. 模型下载](README.md#2-模型下载)
    * [3. 初始化知识库和配置文件](README.md#3-初始化知识库和配置文件)
    * [4. 一键启动](README.md#4-一键启动)
    * [5. 启动界面示例](README.md#5-启动界面示例)
* [联系我们](README.md#联系我们)

## 介绍

🤖️ 一种利用 [langchain](https://github.com/Tian789Gong/SkyOrange-ChatAI.git)
思想实现的基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。


✅ 依托于本项目支持的开源 LLM 与 Embedding 模型，本项目可实现全部使用**开源**模型**离线私有部署**。与此同时，本项目也支持
OpenAI GPT API 的调用，并将在后续持续扩充对各类模型及模型 API 的接入。

⛓️ 本项目实现原理如下图所示，过程包括加载文件 -> 读取文本 -> 文本分割 -> 文本向量化 -> 问句向量化 ->
在文本向量中匹配出与问句向量最相似的 `top k`个 -> 匹配出的文本作为上下文和问题一起添加到 `prompt`中 -> 提交给 `LLM`生成回答。

![实现原理图](img/langchain+chatglm.png)

从文档处理角度来看，实现流程如下：

![实现原理图2](img/langchain+chatglm2.png)

🚩 本项目未涉及微调、训练过程，但可利用微调或训练对本项目效果进行优化。




## 解决的痛点

该项目是一个可以实现 __完全本地化__推理的知识库增强方案, 重点解决数据安全保护，私域化部署的企业痛点。
本开源方案采用```Apache License```，可以免费商用，无需付费。

我们支持市面上主流的本地大语言模型和Embedding模型，支持开源的本地向量数据库。

## 快速上手

### 一、安装项目依赖
#### 1、确认python版本

```shell
# 确认Python 的版本，应该是Pthon3.8~3.11
python3 --version

# 如果版本符合要求就可以直接跳到下一步：2、安装项目依赖；版本不符合要求进行以下步骤更新python版本
# 创建python 3.11虚拟环境
conda create -n env_name python=3.11

# 激活conda环境
conda activate env_name

# 更新pip
pip install --upgrade pip

# 关闭环境
conda deactivate

# 删除环境
conda env remove -n env_name
```

#### 2、安装项目依赖
##### 2.1、安装NVIDIA驱动
```shell
# 更新系统包索引
sudo apt-get update

# 安装NVIDIA驱动
sudo ubuntu-drivers autoinstall

# 重启系统，重启之后可能会让你重新连接服务器，这里正常连接服务器就可以。
sudo reboot

# 重启成功之后，验证NVIDIA驱动是否正确安装并加载
nvidia-smi 

```
##### 2.2、安装项目依赖文件里的依赖环境
```shell
# 拉取项目仓库
git clone https://github.com/Tian789Gong/SkyOrange-ChatAI.git

# 进入项目目录
cd SkyOrange-ChatAI

# 安装全部依赖，依此执行以下3个运行requirements.txt依赖文件的命令
pip install -r requirements.txt

# 阿里云部署如果出现报错：升级 setuptools 和 wheel 后再重新执行安装依赖的命令
pip install setuptools_scm

# 继续安装剩下的依赖
pip install -r requirements_api.txt
pip install -r requirements_webui.txt 
```
### 二、模型下载
以LLM模型为：chatglm3-6b，Embedding模型为：bge-large-zh为例，从魔搭社区拉取模型
```shell
# 返回根目录
cd ~
# 创建专门放模型的目录
mkdir -p models

# 进入模型目录  
cd models

# 更新包列表
sudo apt-get update
    
# 安装Git LFS
sudo apt-get install git-lfs

# 初始化Git LFS
git lfs install

# 从魔搭社区克隆chatglm3-6b仓库
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git

# 进入chatglm3-6b目录
cd chatglm3-6b

# 拉取LFS文件
git lfs pull
 
# 列出LFS对象
ls -lh .git/lfs/objects

# 列出LFS文件检查文件的完整性，检查所有文件 都显示 *号就说明文件是完整的。
# 如果没有*号就说明文件不完整，可以删除文件重新克隆模型仓库
git lfs ls-files

# 返回上一级目录
cd ..       
# 从魔搭社区克隆bge-large-zh仓库
git clone https://www.modelscope.cn/AI-ModelScope/bge-large-zh.git
 
# 进入bge-large-zh目录
cd bge-large-zh
  
# 拉取LFS文件
git lfs pull

# 列出LFS对象
ls -lh .git/lfs/objects
 
# 列出LFS文件检查文件的完整性，检查所有文件 都显示 *号就说明文件是完整的。
# 如果没有*号就说明文件不完整，可以删除文件重新克隆模型仓库
git lfs ls-files 
```
### 三、初始化知识库和配置文件
#### 1、初始化知识库
```shell
# 返回根目录
cd ~
# 进入项目目录
cd SkyOrange-ChatAI

# 复制配置示例
python3 copy_config_example.py
```
#### 2、配置文件
##### 2.1配置模型文件
```shell
# 进入配置文件目录
cd configs

# 用vi编辑器打开 model_config.py文件
vi model_config.py

按下i键进入插入模式
# 修改模型文件目录路径
MODEL_ROOT_PATH = "/root/models"

# 修改选用的 Embedding 名称
EMBEDDING_MODEL = "bge-large-zh"

# 修改MODEL_PATH = {}中的embed_model 模型路径
"bge-large-zh": "/root/models/bge-large-zh"

# 修改MODEL_PATH = {}中的llm_model 模型路径
"chatglm3-6b": "/root/models/chatglm3-6b"

# 按下esc键退出 插入模式，再输入以下保存并退出命令
:wq
```
##### 2.2配置服务文件/重建数据库
```shell
# 进入配置文件目录
cd configs

#用vi编辑器打开 server_config.py文件
vi server_config.py

按下i键进入插入模式
# 在FSCHAT_MODEL_WORKERS = {} 中修改"default" 中的以下键值对
# 使用的GPU ID，以str的格式指定，如"0,1" 有两块GPU就是填写"0,1"，有一块GPU就填写"0"
# 具体情况以你的服务器配置为准。以下以gpu数量为1，显存大小32G为例子
gpus="0" 
num_gpus= 1, 
max_gpu_memory="16GiB"

# 按下esc键退出 插入模式，再输入以下保存并退出命令
:wq
# 返回根目录
cd ~

# 进入项目目录
cd SkyOrange-ChatAI

# 重新创建数据库
python3 init_database.py --recreate-vs

```
### 四、一键启动

按照以下命令启动项目

```shell
# 使用启动文件startup.py进行一键启动
python3 startup.py -a
```

### 五. 启动界面示例

如果正常启动，你将能看到以下界面

1. FastAPI Docs 界面

![](img/fastapi_docs_026.png)

2. Web UI 启动界面示例：

- Web UI 对话界面：

![img](img/LLM_success.png)

- Web UI 知识库管理页面：

![](img/init_knowledge_base.jpg)


---

## 联系我们

🎉 SkyOrange-ChatAI 项目交流QQ群，如果你也对本项目感兴趣，欢迎加入群聊参与讨论交流。

### 交流QQ群

![](img/official_account_qr.png)

🎉 SkyOrange-ChatAI 项目官方QQ群，欢迎扫码进群。
