import pandas as pd
import numpy as np
import transformers
import datasets
import evaluate
from tqdm import tqdm

import os
import torch
# import wandb
from torch import FloatTensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Sequence, Tuple, List

import pytorch_lightning as pl

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

# os.environ["WANDB_PROJECT"]="dnabert_finetuning"
# os.environ["WANDB_LOG_MODEL"]="all"

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tqdm.pandas()
metric = evaluate.load('seqeval')

def read_data(X, y):
    X = X.apply(lambda seq: seq.split(' '))
    y = y.apply(lambda lab: np.reshape(np.array(lab.split(' ')), (-1, 36)))
    return X, y


def compute_metrics(p):
    predictions, labels = p
    predictions = (predictions > 0.5).float()

    # Remove ignored index (special tokens)
    true_predictions = []
    true_labels = []
    
    for p, l in zip(predictions, labels):
        if np.all(l == -100):
            continue
        else:
            true_predictions.append(p)
            true_labels.append(l)

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    model_max_length = 512
    multilabel_length = 36

    def __init__(self, seq, tags, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        self.text = seq

        output = tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            max_length=model_max_length,
            truncation=True,
            is_split_into_words=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        
        labels = []
        for idx in tqdm(range(self.input_ids.shape[0])):
            word_ids = output.word_ids(batch_index=idx)
            tok_lab = []
            for wid in word_ids:
                if wid == None:
                    tok_lab.append(np.full(self.multilabel_length, -100, dtype=float))
                else:
                    tok_lab.append(tags[idx][wid].astype(float))
            
            labels.append(np.array(tok_lab))
        
        self.labels = np.array(labels)
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:        
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels=self.labels[i])

    
class SupervisedDataModule(pl.LightningDataModule):
    
    def __init__(self, x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer, batch_size=16, max_token_len=512):
        super().__init__()
        
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.train_dataset = SupervisedDataset(seq=self.tr_text, tags=self.tr_label, tokenizer=self.tokenizer)
        self.val_dataset = SupervisedDataset(seq=self.val_text, tags=self.val_label, tokenizer=self.tokenizer)
        self.test_dataset = SupervisedDataset(seq=self.test_text, tags=self.test_label, tokenizer=self.tokenizer)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True , num_workers=4)

    def val_dataloader(self):
        return DataLoader (self.val_dataset, batch_size= 16)

    def test_dataloader(self):
        return DataLoader (self.test_dataset, batch_size= 16)
    

class MultiLabelTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[FloatTensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            logging.info(f"Using multi-label classification with class weights", class_weights)
        self.loss_fct = BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        
        try:
            loss = self.loss_fct(outputs.logits.view(-1, model.num_labels), labels.view(-1, model.num_labels))
        except AttributeError:  # DataParallel
            loss = self.loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1, model.num_labels))

        return (loss, outputs) if return_outputs else loss
    
    
if __name__ == "__main__":
    
    data_dir = "/Users/stephenlu/Documents/ml/biocomp/dnabert2/data/atac-seq/"
    model_path = "models/DNA_bert_6"
    model_name = "multilabel_ner_on_atac_seq"
    
    X_train = pd.read_csv(data_dir + "sample_train_seq.tsv", header=None, index_col=None)[0]
    y_train = pd.read_csv(data_dir + "sample_train_label.tsv", header=None, index_col=None)[0]
    X_val = pd.read_csv(data_dir + "sample_train_seq.tsv", header=None, index_col=None)[0]
    y_val = pd.read_csv(data_dir + "sample_train_label.tsv", header=None, index_col=None)[0]
    X_test = pd.read_csv(data_dir + "sample_train_seq.tsv", header=None, index_col=None)[0]
    y_test = pd.read_csv(data_dir + "sample_train_label.tsv", header=None, index_col=None)[0]
    
    X_train, y_train = read_data(X_train, y_train)
    X_val, y_val = read_data(X_val, y_val)
    X_test, y_test = read_data(X_test, y_test)
    
    batch_size = 16
    cache_dir = None
    model_max_length = 512
    multilabel_length = 36
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    
    data_module = SupervisedDataModule(
        list(X_train), y_train,
        list(X_val), y_val,
        list(X_test), y_test,
        tokenizer, 
        batch_size, 
        model_max_length
    )

    data_module.setup()
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=multilabel_length
    )

    args = TrainingArguments(
        model_name,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=20,
        push_to_hub=False,
        # report_to="wandb",
        # run_name="superklass-classifier-top-7"
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = MultiLabelTrainer(
        model,
        args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.evaluate(data_module.test_dataset)