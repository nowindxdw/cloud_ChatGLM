# cloud_ChatGLM
选择 ChatGLM-6B 的理由：
国产，ChatGLM 出自于清华系，当前最大的模型为 130B
130B 的 ChatGLM 综合性能略高于 ChatGPT3，而 ChatGPT3 以上版本都不开源
有适合单机部署的 6B 开源版本。这种硬件要求下，一张 16G 显存的显卡就可以顺滑的跑起来。
已有一定的生态环境，未来前景不错。

官方链接： https://github.com/THUDM/ChatGLM-6B

购买 GPU 云服务器
登录腾讯云，https://cloud.tencent.com/
GPU机型 T4 显卡
最便宜的竞价实例：

选择操作系统，这里选择 Windows。并选择 Windows Server2002 64 位中文版的镜像。
设置存储，SSD云硬盘，单盘模式，75G

下一步：
安全组，注意要打开 3389，否则不能连接远程桌面。如果购机完成后，还是无法连接远程桌面，可以使用腾讯云提供的检查功能，然后在检查结果中开启端口。

设置主机名字和密码


连接后操作：
### 1.安装GPU驱动：
https://www.nvidia.com/Download/Find.aspx
找到 T4 对应的驱动12.0

### 2.安装 CUDA：
https://developer.nvidia.com/cuda-downloads

然后再下载 cudnn, 找到和 cuda配套的

https://developer.nvidia.com/rdp/cudnn-download

无注册账号的可从百度云下载：
cuda11.x版本：
链接：https://pan.baidu.com/s/15_tFzQetWhhOZjKRS9EVuQ?pwd=CD11
提取码：CD11
cuda12.x版本：
链接：https://pan.baidu.com/s/1yO3ffHufu5tMR-1ViK9EHQ?pwd=CD12

### 3.安装 anaconda，python环境
https://www.anaconda.com/blog/individual-edition-2021-05
安装  Anaconda For Windows Server 2022

### 4. 安装 pytorch
(注意不要使用清华源，默认是CPU版本)：
https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

### 5 部署 ChatGLM-6B
下载项目程序包
从 GitHub 下载项目程序包，https://github.com/THUDM/ChatGLM-6B
下载后解压到本地目录，如 C:\ChatGLM\ChatGLM-6B-main
下载模型包 chatglm，https://huggingface.co/THUDM/chatglm-6b/tree/main

huggingface 里不能打包下载，只能一个个下载（因为没有找到打包下载的地方），下载到 C:\chatglm-6b。

8 个模型文件（1G 以上的那 8 个）不用在 huggingface 里下载，从这里下载：https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/


### 6 运行网页版 Demo
pip install -r .\requirements.txt   -i https://pypi.tuna.tsinghua.edu.cn/simple
修改模型路径，编辑 web_demo.py，修改路径为模型包保存的目录

model_path= "C:\\chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

执行如下命令，运行网页版本的 demo，如下
python web_demo.py

### 7.保存镜像
这个 GPU 云服务器的方案是按时间计费的，服务器空闲时间也是计费的，即使关机也不会停止计费。如要停止计费，必须将服务器和云盘都销毁。一旦销毁后，下次还想再使用 ChatGLM 就只能重复以上繁琐的步骤，至少需要 2 个小时。
因此，我们可以利用腾讯提供的 80G 免费快照空间。
当不再需要运行 ChatGLM 时，可以将当前的服务器和云盘保存为镜像和快照，然后销毁相应资源。



## 基于 P-Tuning 微调 ChatGLM-6B
ChatGLM-6B 环境已经有了，接下来开始模型微调，这里我们使用官方的 P-Tuning v2 对 ChatGLM-6B 模型进行参数微调，P-Tuning v2 将需要微调的参数量减少到原来的 0.1%，再通过模型量化、Gradient Checkpoint 等方法，最低只需要 7GB 显存即可运行。

下载 GIT：
Git - Downloading Package (git-scm.com)
安装依赖
bash复制代码# 运行微调需要 4.27.1 版本的 transformers

pip install rouge_chinese nltk jieba datasets  -i https://pypi.tuna.tsinghua.edu.cn/simple


数据集
分别保存为 train.json 和 dev.json，放到 ptuning 目录下，实际使用的时候肯定需要大量的训练数据。

参数调整
修改 train.sh 和 evaluate.sh 中的 train_file、validation_file 和 test_file 为你自己的 JSON 格式数据集路径，并将 prompt_column 和 response_column 改为 JSON 文件中输入文本和输出文本对应的 KEY。可能还需要增大 max_source_length 和 max_target_length 来匹配你自己的数据集中的最大输入输出长度。并将模型路径 THUDM/chatglm-6b 改为你本地的模型路径。

### 1、train.sh 文件修改

PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --train_file train.json \
    --validation_file dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path C:/chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN


train.sh 中的 PRE_SEQ_LEN 和 LR 分别是 soft prompt 长度和训练的学习率，可以进行调节以取得最佳的效果。P-Tuning-v2 方法会冻结全部的模型参数，可通过调整 quantization_bit 来被原始模型的量化等级，不加此选项则为 FP16 精度加载。
### 2、evaluate.sh 文件修改

PRE_SEQ_LEN=32
CHECKPOINT=adgen-chatglm-6b-pt-32-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file dev.json \
    --test_file dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path c:/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN


CHECKPOINT 实际就是 train.sh 中的 output_dir。
### 3 训练
bash train.sh
### 4 推理
bash evaluate.sh

### 5部署微调后的模型

这里我们先修改 web_demo.sh 的内容以符合实际情况，将 pre_seq_len 改成你训练时的实际值，将 THUDM/chatglm-6b 改成本地的模型路径。
这里使用ptuning 文件夹下的 web_demo.sh 部署

PRE_SEQ_LEN=32
CUDA_VISIBLE_DEVICES=0 python web_demo.py \
    --model_name_or_path C:\\chatglm-6b \
    --ptuning_checkpoint output/adgen-chatglm-6b-pt-32-2e-2/checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN
    
    




