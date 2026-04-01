from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import torch

# Load dataset
df = pd.read_csv("dataset.csv")

# Example format:
# ref | hyp | score (0-1)

class EITDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ref = str(self.df.iloc[idx]["ref"])
        hyp = str(self.df.iloc[idx]["hyp"])
        score = float(self.df.iloc[idx]["score"])

        encoding = self.tokenizer(
            ref,
            hyp,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = torch.tensor(score, dtype=torch.float)

        return item

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1
)

dataset = EITDataset(df, tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()