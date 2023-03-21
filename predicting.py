import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class FineTunedModel:
    def __init__(self, model_name: str, path_to_models: str = "models/", to_cuda: str = True):
        """
        AutoModelFineTuned initializing
        :param model_name: local model name
        :param path_to_models: finetuned model local paths
        """
        self.model_name = model_name
        self.model = AutoModelForImageClassification.from_pretrained(path_to_models + model_name)
        if to_cuda:
            self.model = self.model.to("cuda")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(path_to_models + model_name)
        print(f"{model_name} loaded.")

    def predict(self, image_path: str, to_cuda: str = True) -> int:
        """
        Predicting class label for only one image
        :param image_path: full local path to image
        :param to_cuda: is using cuda
        :return: predicted class label
        """
        image = Image.open(image_path)

        encoding = self.feature_extractor(image.convert("RGB"), return_tensors="pt")
        if to_cuda:
            encoding = encoding.to("cuda")

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()

        return predicted_class_idx

    def predict_images(self, images_names: pd.Series, predicting_data_path: str) -> pd.Series:
        """
        Predicting class labels for pd.Series of images
        :param images_names: names of images to predict
        :param predicting_data_path: local path to dir of images
        :return: predicted class labels
        """
        print(f"{self.model_name} predicting...")

        tqdm.pandas()
        predictions = images_names.progress_apply(
            lambda x: self.predict(predicting_data_path + x))

        predictions = pd.Series(predictions, name='label_id')

        print(f"{self.model_name} inference finished.")
        return predictions


def predict_few_models(model_names: list[str], data_path: str) -> pd.DataFrame:
    """
    Predict class labels using few models
    :param model_names: list of local finetuned model names
    :param data_path: train or test data path
    :return: DataFrame of few model predictions and filename for saving
    """
    print(f"Predicting all models for '{data_path}'...")
    current_images = pd.Series(os.listdir(data_path), name='image_name')

    file_name = f"predictions/{data_path.split('/')[-2]}"
    dictionary = {'image_name': current_images}

    for model_name in model_names:
        current_model = FineTunedModel(model_name)
        labels = current_model.predict_images(current_images, data_path)

        dictionary[model_name] = labels

        file_name = file_name + model_name.split('_')[-1]

    preds = pd.DataFrame(dictionary)
    print(f"All models predicted.")

    return preds


path_to_train_data = "data/train/"
path_to_test_data = "data/test/"
path_to_labels = "data/train_labels.csv"

models_list = ['swin_tiny_finetuned', 'vit_base_finetuned', 'convnext_base_finetuned',
               'beit_base_10epochs', 'resnet_50_30epochs']

# Predicting train data for analyze and boosting
train_preds = predict_few_models(models_list, path_to_train_data)

train_preds = train_preds.sort_values('image_name')
train_preds['label_id'] = pd.read_csv(path_to_labels, sep='\t').sort_values('image_name')['label_id'].values

train_preds.to_csv(f"predictions/Train_{len(models_list)}_predictions.csv", index=False, sep='\t')

# Predicting test data for voting and boosting
test_preds = predict_few_models(models_list, path_to_test_data)

test_preds.to_csv(f"predictions/Test_{len(models_list)}_predictions.csv", index=False, sep='\t')
