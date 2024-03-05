# NLP2024-tutorial-3
NLP2024 チュートリアル３ 作って学ぶ日本語大規模言語モデル - 環境構築手順とソースコード

# NLP2024-tutorial-3

```Shell
pip install torch
pip install transformers fugashi unidic-lite
pip install accelerate safetensors
```

```Shell
pip install bitsandbytes
```


```Python
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name()
```

```Python
import torch
torch.backends.mps.is_available()
```

```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_name = "llm-jp/llm-jp-1.3b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.pad_token_id)
print(pipe("語りえぬものについては、", max_length=128))
```

```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
model_name = "llm-jp/llm-jp-13b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.pad_token_id)
print(pipe("語りえぬものについては、", max_length=128))
```

```Shell
git clone https://github.com/llm-jp/llm-jp-eval.git
cd llm-jp-eval
cp configs/config_template.yaml configs/config.yaml
python -m venv venv
source venv/bin/activate
pip install -e .
```

```Shell
python scripts/evaluate_llm.py torch_dtype=fp32 \
target_dataset="[jnli]" \
metainfo.max_num_samples=-1 \
wandb.run_name=llm-jp-1.3b-v1.0_fp32_jnli \
model.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0 \
tokenizer.pretrained_model_name_or_path=llm-jp/llm-jp-1.3b-v1.0
```

```Python
import json
import sys
from datasets import load_dataset
dataset = load_dataset('mc4', 'ja', split='train', streaming=True)
with open("mc4-ja-10k.jsonl", "w", encoding="utf8") as fout:
    count = 0
    for doc in dataset:
        json.dump(doc, fout, ensure_ascii=False)
        print(file=fout)
        count += 1
        if count == 10000:
            break
```
