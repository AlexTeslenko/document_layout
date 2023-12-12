from typing import Dict

import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments

import utils
from data import datamodule
from model import hfmodel


class Training:
    def __init__(self, cfg: Dict) -> None:

        self.training_args = TrainingArguments(
            output_dir=cfg["train"]["output_dir"],
            num_train_epochs=cfg["train"]["num_train_epochs"],
            per_device_train_batch_size=cfg["train"]["train_batch_size"],
            per_device_eval_batch_size=cfg["train"]["eval_batch_size"],
            warmup_steps=cfg["train"]["warmup_steps"],
            weight_decay=cfg["train"]["weight_decay"],
            logging_dir=cfg["train"]["logging_dir"],
            logging_steps=cfg["train"]["logging_steps"],
            evaluation_strategy=cfg["train"]["evaluation_strategy"],
            save_strategy=cfg["train"]["save_strategy"],
            load_best_model_at_end=cfg["train"]["load_best_model_at_end"],
        )

        self.model_name = cfg["model"]["model_name"]
        self.num_classes = cfg["model"]["num_classes"]
        self.train_data_path = cfg["dataset"]["train_data_path"]
        self.test_data_path = cfg["dataset"]["test_data_path"]
        self.val_split = cfg["dataset"]["val_split"]
        self.num_data_columns = cfg["dataset"]["num_data_columns"]
        self.tokenizer_name = cfg["dataset"]["tokenizer_name"]
        self.target_column = cfg["dataset"]["target_column"]
        self.text_columns = cfg["dataset"]["text_columns"]
        self.bool_columns = cfg["dataset"]["bool_columns"]

        self.model = None
        self.dataset = None
        self.trainer = None
        self.metric = evaluate.load(cfg["evaluate"]["metric"])

    def setup(self) -> None:

        self.model = hfmodel.HuggingFaceModel(
            model_name=self.model_name, num_classes=self.num_classes
        )
        self.dataset = datamodule.DataModule(
            train_data_path=self.train_data_path,
            test_data_path=self.test_data_path,
            val_split=self.val_split,
            num_data_columns=self.num_data_columns,
            tokenizer_name=self.tokenizer_name,
            num_classes=self.num_classes,
            target_column=self.target_column,
            text_columns=self.text_columns,
            bool_columns=self.bool_columns
        )
        self.dataset.setup()

        self.trainer = Trainer(
            model=self.model.model,
            args=self.training_args,
            tokenizer=self.dataset.tokenizer,
            train_dataset=self.dataset.data_train_val["train"],
            eval_dataset=self.dataset.data_train_val["test"],
            compute_metrics=self.compute_metrics,
        )

    def start(self) -> None:
        self.trainer.train()

    def eval(self) -> None:
        self.model.model.eval()
        prediction_output = self.trainer.predict(self.dataset.data_test)
        print(prediction_output)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

