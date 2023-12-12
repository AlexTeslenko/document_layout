from typing import Optional, List

import pandas as pd
from datasets import ClassLabel, Dataset
from transformers import AutoTokenizer, BatchEncoding


class DataModule:
    def __init__(
        self,
        train_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        val_split: float = 0.2,
        num_data_columns: int = 10,
        tokenizer_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        target_column: str = "Label",
        text_columns: List = ["Text"],
        bool_columns: List = []
    ) -> None:

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.num_data_columns = num_data_columns
        self.val_split = val_split
        self.num_classes = num_classes
        self.target_column = target_column
        self.text_columns = text_columns
        self.bool_columns = bool_columns

        self.data_train_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self) -> None:
        if self.train_data_path:
            self.data_train_val = self.prepare_data(
                data_path=self.train_data_path, training_mode=True
            )
        if self.test_data_path:
            self.data_test = self.prepare_data(
                data_path=self.test_data_path, training_mode=False
            )

    def prepare_data(self, data_path: str, training_mode: bool) -> Dataset:
        df = pd.read_csv(data_path, encoding_errors="ignore")
        df = df.iloc[:, 0:self.num_data_columns].dropna()
        df = self.text_features_engineering(df=df)
        dataset = self.convert_to_huggingface(df=df)
        dataset = dataset.map(self.tokenize, batched=True)
        if training_mode:
            dataset = dataset.train_test_split(
                test_size=self.val_split,
                shuffle=True,
                stratify_by_column=self.target_column.lower(),
            )

        return dataset

    def convert_to_huggingface(self, df: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(df)
        title_class = ClassLabel(num_classes=self.num_classes)
        dataset = dataset.cast_column(self.target_column, title_class)
        dataset = dataset.rename_column(self.target_column, self.target_column.lower())

        return dataset

    def tokenize(self, examples) -> BatchEncoding:
        return self.tokenizer(
            examples["updatedText"], padding="max_length", truncation=True
        )

    def text_features_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df["updatedText"] = ""

        for i, row in df.iterrows():
            text = ""
            for text_clmn in self.text_columns:
                text += " " + row[text_clmn]
            for bool_clmn in self.bool_columns:
                if row[bool_clmn]:
                    text += " " + bool_clmn

            df.loc[i, "updatedText"] = text

        return df
