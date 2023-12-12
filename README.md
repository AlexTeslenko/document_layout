## Layou Analysis Project

The task is to build machine learning model that is capable of classifying sections as titles and non-titles.

##### To start training and evaluation run the command:

    make setup run

##### Explanation of the solution:

As initial baseline the simple Support Vector Machine model was train. Only text column was used with tf-idf data encoding. **The accuracy of the basileine model is 0.87**.

After further task and data analysis I decided to fine-tune one of the transformer models(BERT).
Transformer NNs show state-of-the-art results in NLP tasks, and fine-tuning allows to get high metrics with limited amount of task-specific dataset and time.
Huggingface was chosen as framework model fine-tuning, as it allows fast prototyping and productionisation of models.
For current iteration only textual and boolena features was used.
Boolena features was added to text column also as text.
For example if row "Employee Involment" has attribut isBold=True, new row is created:
"Employee Involment isBold"
New row are then passed through tokenizer. And tokenized input is then used to train a model.

As dataset is imbalanced, the data with "Title" class was upsampled for one of the iterations of model training experiments. But it didn't imporve the accuracy of the model. So further investigation is required.

##### Results:

As example distilbert-base-uncased(https://huggingface.co/distilbert-base-uncased) model was finetuned for 5 epochs for classification task.
**The accuracy achieved on test dataset: 0.977**

##### Further improvements:

Current approach may be improved in several directions based on business goal(e.g decrease latency of the model, improve accuracy, decrease costs of training or inference of the model).
Current version of the repo is designed in the way which allows quickly train and test various models from Huggingface, change hyperparameters, add new features and data, train for more epochs, just changing config file.
Current version of the repo works only with textual and bollean features. In order to incorporate numerical features more work must be done on data processing and model sides.
As upsampling of minor class didn't improve the accuracy of the model, further investigation is required to overcome imbalance data problem.
Unit and integration tests must be added, and documnetation for the codebase.

##### Time spent:

About 6-8 hours were spent on intialial task and data analysis and training of the first model.
About 6-8 hours we spent on designing and developing the productoion-ready code.
