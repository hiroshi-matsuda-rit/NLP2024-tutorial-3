# NLP2024-tutorial-3
NLP2024 チュートリアル３: 作って学ぶ日本語大規模言語モデル - 環境構築手順と実験ソースコード  
NLP2024 Tutorial 3: Practicing how to build a Japanese large-scale language model - Environment construction and experimental source codes

## Index
- [環境構築手順 / Environment Construction](#環境構築手順)
  - [For Ubuntu](#for-ubuntu)
  - [For WSL2](#for-wsl2)
  - [For macOS](#for-macos)
- [実験ソースコード / Experimental Source Codes](#実験ソースコード)
  - [Software Installation](#ソフトウェアのインストール)
  - [Inference and Evaluation](#inference-and-evaluation)
  - [Supervised Fine-tuning](#supervised-fine-tuning)
  - [Direct Preference Optimization](#direct-preference-optimization)
  - [Pretraining](#pretraining)

# 環境構築手順
**Environment Construction**

## For Ubuntu

### 前提条件 / Prerequisites
- Hardwares
  - CPU Intel 64bit, RAM >=32GB (>=64GB recommended), Free Disk Space >=200GB
  - GPU RAM >=8GB (>=16GB recommended), Compute Capabilty >=7.0 (>=8.0 recommended)
    - Compute Capability 8.0未満ではbfloat16を使用することができない / Cannot use bfloat16 with Compute Capability below 8.0
    - Compute CapabiltyはHPCシステムズ社の[こちらの一覧表](https://www.hpc.co.jp/product/wp-content/uploads/sites/3/2022/07/GPU-list_A3.pdf)を参照 / Compute Capabilty can be checked in [this table](https://www.hpc.co.jp/product/wp-content/uploads/sites/3/2022/07/GPU-list_A3.pdf).
- Softwares
  - Ubuntu 22.04がクリーンインストールされた状態を想定 / Assuming a clean installation of Ubuntu 22.04
  - 環境構築を行うユーザにsudo権限が付与されていること / The sudo privileges have been granted to the user who will be building the environment.

### gcc-12 installation steps
```Shell
sudo apt update
sudo apt upgrade
sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git
sudo apt install gcc-12 g++-12
sudo ln -s -f /usr/bin/gcc-12 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-12 /usr/bin/g++
```

### nvidia-driver-535 installation steps
`nvidia-smi`が実行できたら既にnvidia-driverがインストールされている。  
If you can run `nvidia-smi`, nvidia-driver is already installed.
```Shell
nvidia-smi
```

nvidia-driver-525未満がインストールされていたら下記で一旦削除。525以上がインストールされていたら以降はスキップしてCUDAのインストールに進む。  
If the installed nvidia-driver version is lower than 525, remove it by following the steps below.
If the nvidia-driver version is 525 or higher is installed, skip the rest and proceed to install CUDA.
```Shell
sudo apt-get --purge remove nvidia-*
sudo apt-get --purge remove cuda-*
```

nvidia-driverをインストールして再起動。  
Install nvidia-driver and reboot.
```Shell
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

再起動したらログインして`nvidia-smi`が動作するか確認。  
After restarting, login and check if `nvidia-smi` works.
```Shell
nvidia-smi
```

nvidia-driverが自動更新されて動作しなくなることがあるので、nano等のエディタで設定ファイルの`Unattended-Upgrade`の値を`"0"`に変更しておく。  
Since nvidia-driver may be updated automatically and stop working, change the value of `Unattended-Upgrade` in the configuration file to `"0"` using an editor such as nano.
```Shell
sudo nano /etc/apt/apt.conf.d/20auto-upgrades
```
```Console
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "0";
```

### CUDA 12.1 installation steps
[公式サイト](https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)にあるrunfileでのインストール手順を実行。  
Execute the installation procedure using the runfile on [the official website](https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local).
```Shell
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

既存のドライバを削除することを推奨されるがContinueを選択。  
Although it is recommended to remove the existing driver, select Continue.
```Console
│ Existing package manager installation of the driver found. It is strongly    │
│ recommended that you remove this before continuing.                          │
│ Abort                                                                        │
│ Continue                                                                     │
```

End User License Agreementについて確認したらacceptを入力。  
After confirming the End User License Agreement, enter accept.
```Console
Do you accept the above EULA? (accept/decline/quit):
accept
```

セットアップオプションを次のように設定してInstallを実行。  
Set the setup options as follows and run Install.
```Console
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 530.30.02                                                           │
│ + [X] CUDA Toolkit 12.1                                                      │
│   [ ] CUDA Demo Suite 12.1                                                   │
│   [ ] CUDA Documentation 12.1                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│   Install                                                                    │
```

インストールが終わったらnvccを実行できるか確認。  
Once the installation is complete, check if you can run nvcc.
```Shell
/usr/local/cuda/bin/nvcc -V
```

## For WSL2

### 前提条件 / Prerequisites
- Harwares
  - Ubuntuの[前提条件](#前提条件--prerequisites)に準じる / See the [Prerequisites](#前提条件--prerequisites) section for Ubuntu
- Softwares
  - Windows11 22H2 or later (Windows10 22H2でも動作可能 / can also operate on Windows10 22H2)
  - WSL2上でUbuntu 22.04がクリーンインストールされた状態を想定 / Assuming a clean installation of Ubuntu 22.04 on WSL2
  - 環境構築を行うユーザにAdministrator権限が付与されていること / The user who will be building the environment must be granted Administrator privileges

### Windows側でNVIDIA Driverをインストール / Install NVIDIA Driver on Windows side
NVIDIAの[ドライバーダウンロードページ](https://www.nvidia.co.jp/Download/index.aspx?lang=jp#)から使用する製品とOSを選択し、ダウンロードタイプは製品ブランチ/Studioを指定して、探すを押下。  
Select the product and OS you are using from the NVIDIA [driver download page](https://www.nvidia.co.jp/Download/index.aspx), specify the Product Branch / Studio as the download type, and press Search.

---
<img width="480" alt="nvidia-driver-download-setting-en" src="https://github.com/hiroshi-matsuda-rit/NLP2024-tutorial-3/assets/40782025/7232bef4-35c3-4ec4-a98e-7536a6503780">

---
<img width="480" alt="nvidia-driver-download-setting-en" src="https://github.com/hiroshi-matsuda-rit/NLP2024-tutorial-3/assets/40782025/6c9f0fab-bb6a-4363-8eae-ffdad235881d">

---

ダウンロードしたファイルを実行してドライバをインストール。  
Run the downloaded file to install the driver.

### WSL2でUbuntu 22.04をインストール/ Install Ubuntu 22.04 with WSL2

#### 管理者権限でPowerShellを起動する / Start PowerShell with administrator privileges

- Windowsボタンを右クリックして`ターミナル（管理者）`を選択するとPowerShellが起動する / Right-click the Windows button and select Terminal (Administrator) to start PowerShell

#### WSL2の更新 / Update WSL2

- PowerShellで次を実行して利用可能なLinuxディストリビューションのリストを表示 / View a list of available Linux distributions by running the following in PowerShell
```Shell
wsl --set-default-version 2
wsl --update
```

#### WSL2上でのUbuntu 22.04のインストール / Installing Ubuntu 22.04 on WSL2

下記を実行してユーザ設定を行います。 / Execute the following to configure the user settings.
```Shell
wsl --install -d Ubuntu-22.04
```
引き続きUbuntu側でnvidia-smiの動作確認を行います。 / Continue to check the operation of nvidia-smi on the Ubuntu side.
```Shell
nvidia-smi
```

### Ubuntu側でのCUDAのインストール / Installing CUDA on Ubuntu side

WSL2上のUbuntuで、Ubuntu編の[gcc等のインストール](#gcc-12-installation-steps)、および、[CUDA 12.1のインストール](#cuda-121-installation-steps)を実施します。  
On Ubuntu on WSL2, perform the steps described in the Ubuntu edition for [gcc-12 installation steps](#gcc-12-installation-steps) and [CUDA 12.1 installation steps](#cuda-121-installation-steps).

### Windowsターミナルのインストール / Installing Windows Terminal
以降の作業と実験の作業性をよくするためWindowsターミナルの利用を推奨します。 / We recommend using Windows Terminal to improve the workability of subsequent work and experiments.  
[Microsoft Store](https://apps.microsoft.com/detail/9n0dx20hk701?rtc=1&activetab=pivot%3Aoverviewtab&hl=ja-jp&gl=JP)からインストールできます。 / It can be installed from [Microsoft Store](https://apps.microsoft.com/detail/9n0dx20hk701?rtc=1&activetab=pivot%3Aoverviewtab).

## For macOS

### 前提条件/ Prerequisites
- Hardwares
  - CPU Apple M1 or later, RAM >=16GB (>=32GB recommended), Free Disk Space >=200GB
- Softwares
  - macOS 13 or later

### Installing Command Line Tools
Command Line Toolsをインストールしていない場合はコンソールアプリで下記を実行。 / If you do not have Command Line Tools installed, run the following in the console app.
```Shell
xcode-select --install
```

### Installing Python3.10.11 and PATH setting
python.orgから[python 3.10.11 macOS 64-bit universal2 installer](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg)をダウンロードして実行。 / Download [python 3.10.11 macOS 64-bit universal2 installer](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg) from python.org and run it.

# 実験ソースコード
**Experimental Source Codes**

## ソフトウェアのインストール

### CUDAの動作確認
- Ubuntu / WSL2
```Shell
/usr/local/cuda/bin/nvcc -V
```

### 環境変数LD_LIBRARY_PATHにCUDAのパスを追加
- Ubuntu / WSL2
```Shell
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64"' >> ~/.bashrc
source ~/.bashrc
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
```Shell
curl https://pyenv.run | bash
```

#### pyenv導入済みの場合
```Shell
cd ~/.pyenv/plugins/python-build/../.. && git pull && cd -
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
  - ~/.bash_profile（zshの場合は ~/.zshrc）に追加
```Shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init --path)"' >> ~/.bash_profile
source ~/.bash_profile
```

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
  - Pythonの対話モードで下記を実行。
```Python
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name()
```
- Mac
  - Pythonの対話モードで下記を実行。
```Python
import torch
torch.backends.mps.is_available()
```

### Transformers＋BERTで動作確認

```Shell
pip install transformers fugashi unidic-lite
```

- Ubuntu / WSL2
  - Pythonの対話モードで下記を実行。
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
  - Pythonの対話モードで下記を実行。
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
  - Pythonの対話モードで下記を実行。
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
  - Pythonの対話モードで下記を実行。
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
  - Pythonの対話モードで下記を実行。
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
  - Pythonの対話モードで下記を実行。
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
  - Pythonの対話モードで下記を実行。
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

#### dataset_dirの設定

- `configs/config.yaml`をエディタで開き、上で確認した`dev/`までのパスを`dataset_dir`の値を次のようにセットする
```yaml
dataset_dir: "jaster/1.2.0/evaluation/dev"
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

< TOKENIZER_PATH=./tmp/tokenizer.model # official llama tokenizer.model
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
