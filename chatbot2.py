import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Define the path to save the model
model_save_path = "./convmodel"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-1_6b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-2-1_6b", torch_dtype="auto")

# Load the DailyDialog dataset
dataset = load_dataset("daily_dialog")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["dialog"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator used for padding and combining the data into batches
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_save_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    trust_remote_code=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
trainer.save_model(model_save_path)

# Inform the user that fine-tuning is complete
print(f"Model fine-tuned and saved to {model_save_path}")
