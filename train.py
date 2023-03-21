import evaluate
import numpy as np
import torch
import pandas as pd
import datasets
import wandb
from datasets import load_metric
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    Grayscale,
    ToTensor,
)

train_labels = pd.read_csv("data/train.csv", sep=",")

## На 1000 классов
classes = 2000
train_labels = train_labels[(train_labels['class_id'] >= classes - 1000) & (train_labels['class_id'] < classes)]
##

model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"
# model_checkpoint = "google/vit-base-patch16-384"
# model_checkpoint = "facebook/convnext-base-224"
# model_checkpoint = "microsoft/resnet-50"
# model_checkpoint = "google/mobilenet_v2_1.0_224"
# model_checkpoint = "microsoft/beit-base-patch16-224-pt22k-ft22k"

size_field = "height"  # height shortest_edge


metric = evaluate.load("matthews_correlation") # accuracy

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

train = train_labels.copy()
path_to_train = "data/images/"
train['path'] = path_to_train + train['image']

features = datasets.Features({
    'image': datasets.Image(decode=True, id=None),
    'label': datasets.ClassLabel(num_classes=1000, id=None)
})

dataset_dict = {
    'image': train['path'].values,
    'label': train['class_id'].values
}

train_dataset = datasets.Dataset.from_dict(dataset_dict, features)
dataset = datasets.DatasetDict({'train': train_dataset})

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(feature_extractor.size[size_field]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(feature_extractor.size[size_field]),
        CenterCrop(feature_extractor.size[size_field]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
).to("cuda")

model_name = f"{classes}_" + model_checkpoint.split("/")[-1].split('-')[0] + "_" + model_checkpoint.split("/")[-1].split('-')[1] + "_checkpoints"

wandb.init(project="clean_code_cup",
           name=model_name)

batch_size = 16
path_to_models = "models/"

args = TrainingArguments(
    f"{path_to_models}{model_name}",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=5e-5,
    num_train_epochs=10,
    gradient_accumulation_steps=3,
    per_device_train_batch_size=batch_size,

    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=50,
    report_to=["wandb"]
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print("Обучаем")

train_results = trainer.train()
# rest is optional but nice to have
wandb.finish()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
