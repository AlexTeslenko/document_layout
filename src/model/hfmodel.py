from transformers import AutoModelForSequenceClassification


class HuggingFaceModel:
    def __init__(
        self, model_name: str = "distilbert-base-cased", num_classes: int = 2
    ) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
