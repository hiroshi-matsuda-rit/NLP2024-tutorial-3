# NLP2024-tutorial-3
NLP2024 チュートリアル３ 作って学ぶ日本語大規模言語モデル - 環境構築手順と実験ソースコード

## Index
- 環境構築手順 (準備中)
  - Ubuntu編
  - WSL2編
  - Mac編
<!--
- [環境構築手順](#環境構築手順)
  - [Ubuntu編](#ubuntu編)
  - [WSL2編](#wsl2編)
  - [Mac編](#mac編)
-->
- [実験ソースコード](#実験ソースコード)
  - [ソフトウェアのインストール](#ソフトウェアのインストール)
  - [Inference and Evaluation](#inference-and-evaluation)
  - [Supervised Fine-tuning](#supervised-fine-tuning)
  - [Pretraining](#pretraining)

# 環境構築手順

## Ubuntu編
Ubuntu 22.04を使用

(準備中)

## WSL2編
Windows11 22H2以降のWSL2上でUbuntu 22.04を使用

(準備中)

## macOS編 
M1以降のMacでmacOS 14以降を使用

(準備中)

# 実験ソースコード

## ソフトウェアのインストール

### CUDAの動作確認
```Shell
/usr/local/cuda/bin/nvcc -V
```

### 環境変数LD_LIBRARY_PATHにCUDAのパスを追加
```Shell
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64"' >> ~/.bashrc
```

### python3でvenvが使える状態かの確認
```Shell
python3 -V
python3 -m venv venv
source venv/bin/activate
deactivate
rm -r venv
```

### pyenv環境の構築

#### pyenv未導入の場合
- Ubuntu / WSL2
```Shell
curl https://pyenv.run | bash
```
- Mac
```Shell
brew install pyenv
```

#### pyenv導入済みの場合
- Ubuntu / WSL2
```Shell
cd ~/.pyenv/plugins/python-build/../.. && git pull && cd -
```
- Mac
```Shell
brew update
brew upgrade pyenv
```

### pyenvのパス追加
- Ubuntu / WSL2
  - ~/.bashrc（zshの場合は ~/.zshrc）に追加
```Shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
source ~/.bashrc
```
- Mac
  - home brewでpyenvを導入した場合はパス設定は不要

### pyenvでPython 3.10.13をインストール
```Shell
pyenv install 3.10.13
```

### 実験ディレクトリとvenv環境の作成・有効化・バージョン確認
```Shell
mkdir my-llm
cd my-llm
pyenv local 3.10.13
python -m venv venv
source venv/bin/activate
which python
python -V
pip -V
```

### PyTorchのインストールと動作確認

```Shell
pip install torch
```

- Ubuntu / WSL2
```Python
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name()
```
- Mac
```Python
import torch
torch.backends.mps.is_available()
```

### Transformers＋BERTで動作確認

```Shell
pip install transformers fugashi unidic-lite
```

- Ubuntu / WSL2
```Python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
model_name = "cl-tohoku/bert-large-japanese-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model = model.to("cuda:0")
mlm = pipeline("fill-mask", model=model, tokenizer=tokenizer, device="cuda:0")
mlm("語りえぬものについては、[MASK]しなければならない。")[:2]
```
- Mac
```Python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
model_name = "cl-tohoku/bert-large-japanese-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model = model.to("mps")
mlm = pipeline("fill-mask", model=model, tokenizer=tokenizer, device="mps")
mlm("語りえぬものについては、[MASK]しなければならない。")[:2]
```

## Inference and Evaluation

### text-generation実験

```Shell
pip install accelerate safetensors bitsandbytes
```

#### 1.3B
- FP32
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_name = "llm-jp/llm-jp-1.3b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.pad_token_id)
print(pipe("語りえぬものについては、", max_length=128))
```
- FP16
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_name = "llm-jp/llm-jp-1.3b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.pad_token_id)
print(pipe("語りえぬものについては、", max_length=128))
```
- BF16 - Ubuntu / WSL2
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_name = "llm-jp/llm-jp-1.3b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.pad_token_id)
print(pipe("語りえぬものについては、", max_length=128))
```

#### 13B
- FP16
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_name = "llm-jp/llm-jp-13b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.pad_token_id)
print(pipe("語りえぬものについては、", max_length=128))
```
- 4bit - Ubuntu / WSL2
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
model_name = "llm-jp/llm-jp-13b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config)
```

### llm-jp-eval

#### インストール

- venv環境に入っている場合はいったん抜ける
```Shell
deactivate
```
- llm-jp-evalのcloneとvenv環境の作成・有効化
```Shell
git clone https://github.com/llm-jp/llm-jp-eval.git
cd llm-jp-eval
cp configs/config_template.yaml configs/config.yaml
python -m venv venv
source venv/bin/activate
pip install -e .
wandb disabled
```

#### jasterのビルドとディレクトリ構成の確認

```Shell
python scripts/preprocess_dataset.py --dataset-name all --output-dir jaster/
ls jaster/
ls jaster/1.2.0/
ls jaster/1.2.0/evaluation
```

#### 精度評価

##### JNLI devセット全件の評価

- FP32
```Shell
python scripts/evaluate_llm.py torch_dtype=fp32 \
  target_dataset="[jnli]" \
  metainfo.max_num_samples=-1 \
  wandb.run_name=llm-jp-1.3b-v1.0_fp32_dev-jnli \
  model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
  tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```
- FP16
```Shell
python scripts/evaluate_llm.py torch_dtype=fp16 \
  target_dataset="[jnli]" \
  metainfo.max_num_samples=-1 \
  wandb.run_name=llm-jp-1.3b-v1.0_fp16_dev-jnli \
  model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
  tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```
- BF16 - Ubuntu / WSL2
```Shell
python scripts/evaluate_llm.py torch_dtype=bf16 \
  target_dataset="[jnli]" \
  metainfo.max_num_samples=-1 \
  wandb.run_name=llm-jp-1.3b-v1.0_bf16_dev-jnli \
  model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
  tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```

##### jaster全データセット先頭100件の評価

- FP32
```Shell
python scripts/evaluate_llm.py torch_dtype=fp32 \
  wandb.run_name=llm-jp-1.3b-v1.0_fp32_dev-all \
  model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
  tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```
- FP16
```Shell
python scripts/evaluate_llm.py torch_dtype=fp16 \
  wandb.run_name=llm-jp-1.3b-v1.0_fp16_dev-all \
  model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
  tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```
- BF16 - Ubuntu / WSL2
```Shell
python scripts/evaluate_llm.py torch_dtype=bf16 \
  wandb.run_name=llm-jp-1.3b-v1.0_bf16_dev-all \
  model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
  tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```

## Supervised Fine-tuning

### インストール

- llm-jp-eval等の環境に入っている場合はいったん抜ける
```Shell
deactivate
cd ..
```
- llm-jp-sftのcloneとvenv環境の作成・有効化
```Shell
git clone https://github.com/llm-jp/llm-jp-sft.git
cd llm-jp-sft
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
wandb disabled
```
- Macでは `pip uninstall bitsandbytes` を行っておく

### jasterの参照

- llm-jp-evalのjasterディレクトリへのsymbolic linkを作成しておく
```Shell
ln -s ../llm-jp-eval/jaster .
```

### Ichikara-instruction公開データのプロンプト化

- 次のページから利用許諾を確認した上で公開データを入手する
  - [LLMのための日本語インストラクションデータ作成プロジェクト](https://liat-aip.sakura.ne.jp/wp/llm%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%A9%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%87%E3%83%BC%E3%82%BF%E4%BD%9C%E6%88%90/)
- `convert_ichikara.py`の作成
```Python
import json
import random
import sys

if __name__ == "__main__":
    inst = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
    records = []
    for f in sys.argv[1:]:
        with open(f, "r", encoding="utf8") as fin:
            for r in json.load(fin):
                records.append({
                    "ID": r["ID"],
                    "text": f'{inst}\n\n### 指示:\n{r["text"]}\n\n### 応答:\n{r["output"]}',
                })
    random.shuffle(records)
    dev_len = len(records) // 10
    dev, train = records[:dev_len], records[dev_len:]
    json.dump(train, sys.stdout, indent=1, ensure_ascii=False)
    json.dump(dev, sys.stderr, indent=1, ensure_ascii=False)
```
- 公開データの変換と出力の確認
```Shell
python convert_ichikara.py Distribution20231115/*.json \
  > jaster/1.2.0/tuning/train/ichikara.json \
  2> jaster/1.2.0/tuning/dev/ichikara.json
head -n 5 jaster/1.2.0/tuning/dev/ichikara.json
```

### LoRA SFT BF16 - Ubuntu / WSL2

```Shell
python train.py \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --data_files `ls jaster/1.2.0/tuning/train/*.json` \
  --use_peft \
  --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
  --output_dir results/llm-jp-1.3b-lora-jaster_ichikara-v1.0
```

### LoRA SFT 4bit - Ubuntu / WSL2

```Shell
python train.py \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 \
  --load_in_4bit True \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --data_files `ls jaster/1.2.0/tuning/train/*.json` \
  --use_peft \
  --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
  --output_dir results/llm-jp-1.3b-lora-jaster_ichikara-v1.0
```

### フルパラメータSFT

- 8GPU構成向けの`configs/accelerate_config_zero3.yaml`の作成
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
deepspeed_config:
  zero_stage: 3
  offload_optimizer_device: none
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
```
- 13Bモデルの8GPUフルパラメータSFT
```Shell
accelerate launch --config_file configs/accelerate_config_zero3.yaml \
  train.py \
  --model_name_or_path llm-jp/llm-jp-13b-v1.0 \
  --tokenizer_name_or_path llm-jp/llm-jp-13b-v1.0 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 --warmup_ratio 0.1 --lr_scheduler_type cosine \
  --bf16 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --data_files `ls jaster/1.2.0/tuning/train/*.json` \
  --output_dir results/llm-jp-13b-full-jaster_ichikara-v1.0
```

## Direct Preference Optimization

### リポジトリのclone

- llm-jp-sft等の環境に入っている場合はいったん抜ける
```Shell
deactivate
cd ..
```
- llm-jp-dpoをclone
```Shell
git clone https://github.com/llm-jp/llm-jp-dpo.git
cd llm-jp-dpo
```

### Ubuntu / WSL2
- poetryでライブラリをインストール
```Shell
poetry install
poetry shell
wandb disabled
```
- accelerate_config/single_gpu.yamlを作成
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
- AccelerateでDPO学習プロセスを起動
```Shell
accelerate launch --config_file accelerate_configs/single_gpu.yaml train.py --model llm-jp/llm-jp-1.3b-v1.0 --per-device-train-batch-size 4 --per-device-eval-batch-size 8
```

#### Mac (パフォーマンスに難があるため改良中です)
- ライブラリのインストールはpoetryではなくpipで行う
```Shell
python -m venv venv
source venv/bin/activate
pip install torch==2.2.0 transformers==4.37.2 trl==0.7.10 peft==0.8.2 datasets==2.16.1 accelerate==0.26.1 wandb
wandb disabled
```
- エディタで`train.py`を開き`main()`の先頭に`torch.dynamo`のエラー対策を追加
```Python
def main():
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
```
- 同様に`AutoModelForCausalLM.from_pretrained()`の`torch_dtype`を`float16`に変更
```Python
        torch_dtype=torch.float16, # bfloat16,
```
- 同様に`TrainingArguments()`から`bf16`の指定をコメントアウト
```Python
        # bf16=True,
```
- PythonでDPO学習プロセスを起動
```Shell
python train.py --model llm-jp/llm-jp-1.3b-v1.0 --per-device-train-batch-size 4 --per-device-eval-batch-size 8
```

## Pretraining

### 環境構築
- llm-jp-dpo等の環境に入っている場合はいったん抜ける
```Shell
deactivate
cd ..
```
- CUDA 11.8向けにMegatron-DeepSpeed環境を構築する
```Shell
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64"' >> ~/.bashrc
source ~/.bashrc
git clone https://github.com/microsoft/Megatron-DeepSpeed
cd Megatron-DeepSpeed
mkdir -p tmp
python3 -m venv venv
source venv/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install "pip>=23.1" "setuptools>=65.5.0" wheel pybind11 six regex nltk numpy deepspeed==0.12.2 einops tensorboard transformers sentencepiece "protobuf<3.21.0"
```

#### apexのインストール
- [Installation - From Source - Linux](https://github.com/NVIDIA/apex#linux) の手順を pip>=23.1 の前提で進める
- エラーにハマりやすいので以下の点に注意して作業を行う
  - 本来10分ほどかかるはずのビルド処理がすぐに終わる場合は*.soのコンパイルがスキップされている
  - build/lib.linux-x86_64-cpython-310/apex_C.cpython-310-x86_64-linux-gnu.so がなければビルド失敗
```Shell
git clone https://github.com/NVIDIA/apex -b 23.08 ; cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
```

#### FlashAttention2のインストール
- ninjaのバージョンが 1.11.1 か確認する
```Shell
ninja --version
```
- バージョン上限を指定してflash-attnをインストール
```Shell
pip install "flash-attn<2.4.0" --no-build-isolation
```

### トークナイザの準備
- llm-jp-tokenizer v2.1 SentencePieceモデルファイルのダウンロード
```Shell
curl -O -L https://github.com/llm-jp/llm-jp-tokenizer/raw/main/models/ver2.1/code10k_en20k_ja30k.ver2.1.model
```

### 事前学習データの準備
- 次の内容で`download_mc4_ja.py`を作成
```Python
import json
import sys
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset('mc4', 'ja', split='train', streaming=True)
    limit = int(sys.argv[1])
    count = 0
    for doc in dataset:
        json.dump(doc, sys.stdout, ensure_ascii=False)
        print()
        count += 1
        if count == limit:
            break
```
- `download_mc4_ja.py`を実行してmC4の日本語パートから先頭1万件を`mc4-ja-10k.jsonl`に保存
```Shell
python download_mc4_ja.py 10000 > mc4-ja-10k.jsonl
```
- データセットをビルドして作成されたファイルを確認
```Shell
python tools/preprocess_data.py \
  --input ./mc4-ja-10k.jsonl \
  --output-prefix dataset/mc4-ja-10k \
  --tokenizer-model ./code10k_en20k_ja30k.ver2.1.model \
  --append-eod \
  --tokenizer-type SentencePieceTokenizer \
  --dataset-impl mmap --workers 8
ls -l dataset/mc4-ja-10k*
```

### 事前学習スクリプトの準備
- サンプルの`pretrain_llama2_distributed.sh`をコピー
```Shell
cp examples_deepspeed/pretrain_llama2_distributed.sh .
chmod +x pretrain_llama2_distributed.sh
```
- `./pretrain_llama2_distributed.sh`を編集して次の行を変更
  - `<`の行を`>`の行の内容に置き換える
```
< DATASET_1="./tmp/data/bookcorpus_train_1m_text_sentence"
> DATASET_1="./dataset/mc4-ja-10k_text_document"

< TOKENIZER_PATH=./tmp/tokenizer.model # offical llama tokenizer.model
> TOKENIZER_PATH=./code10k_en20k_ja30k.ver2.1.model
> export NCCL_IB_GID_INDEX=3
> export NCCL_IB_TC=106

<        --tokenizer-type GPTSentencePieceTokenizer \
>        --tokenizer-type SentencePieceTokenizer \
```
- pretrain_gpt.pyの最後から2行目のデフォルトトークナイザの指定をコメントアウト
```Python
             # args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
```

### 事前学習の実行
```Shell
./pretrain_llama2_distributed.sh
```

以上
