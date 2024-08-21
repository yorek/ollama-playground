import sys
import logging
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Model checkpoint to fine-tune
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"

# Load your dataset from CSV into a pandas DataFrame
df_test = pd.read_csv("./content/test.csv")

# Prepare your dataset
dataset_test = Dataset.from_pandas(df_test)

# Display information about the dataset
print("===== TEST SET INFO =====")
print(df_test.count())
print(dataset_test.info)

# Load your dataset from CSV into a pandas DataFrame
df_train = pd.read_csv("./content/train.csv")

# Prepare your dataset
dataset_train = Dataset.from_pandas(df_train)

# Display information about the dataset
print("===== TRAIN SET INFO =====")
print(df_train.count())
print(dataset_train.info)

# Model loading arguments
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    #attn_implementation="flash_attention_2",  # Flash Attention support usable only with GPU
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map=None
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Tokenizer configuration
tokenizer.model_max_length = 2048  # Set maximum sequence length
tokenizer.pad_token = tokenizer.unk_token  # Use unk as padding token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# Training hyperparameters
training_config = {
    "bf16": False,  # Use mixed precision
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./output",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2
}

# PEFT (LoRA) configuration
peft_config = {
    "r": 16,  # LoRA rank
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None
}

# Create TrainingArguments and LoraConfig objects
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    max_seq_length=2048,
    dataset_text_field="messages",
    tokenizer=tokenizer,
    packing=False
)

# Train the model
train_result = trainer.train()

# Log and save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Adjust tokenizer padding side for evaluation
tokenizer.padding_side = 'left'

# Evaluate the model
metrics = trainer.evaluate()

# Log and save evaluation metrics
metrics["eval_samples"] = len(dataset_test)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

trainer.save_model(train_conf.output_dir)

trainer.model.save_pretrained("mymodel")
trainer.tokenizer.save_pretrained("mymodel")