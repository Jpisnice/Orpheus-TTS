from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
import torch
import yaml
import wandb

config_file = "config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]
model_name = config["model_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_percent = config["batch_percent"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
learning_rate = config["learning_rate"]

grad_accum = config.get("gradient_accumulation_steps", 1)
max_steps = config.get("max_steps", -1)
resume_flag = config.get("resume_from_checkpoint", False)
save_total_limit = config.get("save_total_limit", None)

gradient_ckpt = config.get("gradient_checkpointing", False)
torch_empty_cache_steps = config.get("torch_empty_cache_steps", None)
optim = config.get("optim", "adamw_torch")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = pad_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
)

if gradient_ckpt:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

ds = load_dataset(dsn, split="train")
batch_size = max(1, int(len(ds) * batch_percent / 100))

def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token)
    labels = padded.clone()
    labels[labels == pad_token] = -100
    attention_mask = (padded != pad_token).long()
    return {"input_ids": padded, "labels": labels, "attention_mask": attention_mask}

wandb.init(project=project_name, name=run_name)

training_args = TrainingArguments(
    output_dir=f"./{base_repo_id}",
    overwrite_output_dir=True,

    num_train_epochs=epochs,
    max_steps=max_steps,

    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,

    learning_rate=learning_rate,
    logging_steps=1,

    bf16=use_bf16,
    fp16=not use_bf16,

    gradient_checkpointing=gradient_ckpt,

    optim=optim,
    torch_empty_cache_steps=torch_empty_cache_steps,

    save_steps=save_steps,
    save_total_limit=save_total_limit,

    report_to="wandb",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=collate_fn,
)

trainer.train(resume_from_checkpoint=True if resume_flag else None)
