# T5-training-on-ELI5

The ELI5 dataset tries to address these by creating a long form question answering
dataset. If a human tries to explain something to a five-year-old, the answer would be complete without
any factual and informative assumptions. So, ELI5 chose the subreddit forum named, Explain Like I’m Five
(ELI5) which regards comprehension quality of the answer to a question. Now in subreddit forums, this
comprehension quality can be quantified by “upvote” and “downvote” scores. Hence, only the answers
with upvote scores of at least two were chosen. Supporting documents from web data provided by
Common Crawl was also used so that the system can refer to it while generating answers. About 272K
questions were gathered and split the web sources into sentences and measure TFIDF with the question.
The concatenation of the sentences with highest TFIDF similarity with respect to the question becomes
the supporting document. We use T5-small to train the data to build a QA system.

Because the python file assumes that the trained model to Hugginface_Hub, please run before,

```
from huggingface_hub import notebook_login
notebook_login()
```

If there is no intention of pushing the model to the HUB, please change ```push_to_hub``` passed in train function to ```False```.
```
def train(self, Epochs, train_name, validation_name, batch_size, lr, strategy, weight_decay,
                 save_limit, fp16, push_to_hub = True):
```

In the train function, args have been defined as,
```
self.args = Seq2SeqTrainingArguments(
            f"{self.model_name}-finetuned-{self.dataset_name}",
            evaluation_strategy = strategy,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            save_total_limit=save_limit,
            num_train_epochs=Epochs,
            predict_with_generate=True,
            fp16=fp16,
            push_to_hub=push_to_hub,
        )
```

Also please remove the following line in ```train``` function, 

```
self.trainer.push_to_hub()
```


Run the python file on GPU. In the file, edit the following code block,

```
t5_custom = T5_Small_Custom(model_checkpoint= model_checkpoint, dataset_name= dataset_name, metric_name= metric_name, prefixes=prefixes)
t5_custom.train(Epochs = 1, train_name = "train_eli5", validation_name = "validation_eli5", 
                    batch_size = 16, lr = 2e-5, strategy = "epoch" , weight_decay = 0.01, save_limit = 3, fp16 = True)
```

## Results:
##Training Results - 

```
Epochs = 1
Train Loss = 3.964
Validation Loss = 3.754
Rouge1 = 9.6972
Rouge2 = 1.8303
RougeL = 7.8132
RougeLSum = 8.9964
```
##Test Results - 

```
Epochs = 1
Test Loss = 3.75161,
Rouge1: 9.5595,
Rouge2: 1.7879,
RougeL: 7.7084,
RougeLsum': 8.8886,
```


