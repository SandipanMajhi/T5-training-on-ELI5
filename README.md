# T5-training-on-ELI5

The ELI5 dataset tries to create a long form question answering
dataset. If a human tries to explain something to a five-year-old, the answer would be complete without
any factual and informative assumptions. So, ELI5 chose the subreddit forum named, Explain Like I’m Five
(ELI5) which regards comprehension quality of the answer to a question. Now in subreddit forums, this
comprehension quality can be quantified by “upvote” and “downvote” scores. Hence, only the answers
with upvote scores of at least two were chosen. Supporting documents from web data provided by
Common Crawl was also used so that the system can refer to it while generating answers. About 272K
questions were gathered and split the web sources into sentences and measure TFIDF with the question.
The concatenation of the sentences with highest TFIDF similarity with respect to the question becomes
the supporting document. We use T5-small to train the data to build a QA system.

Updates : 

```
10/2/2022 - T5-small trained for 3 epochs  - https://huggingface.co/Sandipan1994/t5-small-finetuned-eli5
10/2/2022 - Experimentation with Generation Parameters - Results Log file uploaded (unprocessed)
```

### Prediction Examples

```
question: How to explain depression to my spouse who believes it's not a real disease and can be controlled by deciding to not be sad. context:  

T5 Answer: "It's not a real disease, but if you aren't sad, you're not going to be sad. If you have depression, your spouse will be able to control it by deciding to not feel sad or not. You'll have to think of it as 'assailable', and you will not be happy with it. I'm not sure what you mean by depression. Basically, depression can be controlled by not being sad because it is not an actual disease. This is why depression is the most common disease in the world. The reason it can happen is that you can feel the pain in your life. It is incredibly difficult to get rid of the symptoms of depression as well. Your spouse has to deal with depression in order to prevent it from happening. There are many ways to explain it to my spouse who thinks it'd be difficult for you to do so is to have the same effect on your brain and your body. When you think about it, the person who is sad and sad is so much more likely to suffer from depression than it would be. So, when you do not think that depression has been triggered by the fact that it has no effect."
```


Please install the following packages, 
```
pip3 install transformers
pip3 install datasets
pip3 install rouge_score
pip3 install transformers
pip3 install datasets
pip3 install rouge_score
pip install 'gem-metrics @ git+https://github.com/GEM-benchmark/GEM-metrics.git' bert_score
```


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

### Updates - 

Training Results - 

```
Epochs = 3
Train Loss = 3.935500
Validation Loss = 3.727522
Rouge1 = 9.944000
Rouge2 = 1.908000
RougeL = 8.014500
RougeLSum = 9.227500
```

## Initial Results - 

Training Results - 

```
Epochs = 1
Train Loss = 3.964
Validation Loss = 3.754
Rouge1 = 9.6972
Rouge2 = 1.8303
RougeL = 7.8132
RougeLSum = 8.9964
```
Test Results - 

```
Epochs = 1
Test Loss = 3.75161
Rouge1 = 9.5595
Rouge2 = 1.7879
RougeL = 7.7084
RougeLsum = 8.8886
```
## ELI5 citation -

```
@inproceedings{fan-etal-2019-eli5,
    title = "{ELI}5: Long Form Question Answering",
    author = "Fan, Angela  and
      Jernite, Yacine  and
      Perez, Ethan  and
      Grangier, David  and
      Weston, Jason  and
      Auli, Michael",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1346",
    doi = "10.18653/v1/P19-1346",
    pages = "3558--3567",
    abstract = "We introduce the first large-scale corpus for long form question answering, a task requiring elaborate and in-depth answers to open-ended questions. The dataset comprises 270K threads from the Reddit forum {``}Explain Like I{'}m Five{''} (ELI5) where an online community provides answers to questions which are comprehensible by five year olds. Compared to existing datasets, ELI5 comprises diverse questions requiring multi-sentence answers. We provide a large set of web documents to help answer the question. Automatic and human evaluations show that an abstractive model trained with a multi-task objective outperforms conventional Seq2Seq, language modeling, as well as a strong extractive baseline.However, our best model is still far from human performance since raters prefer gold responses in over 86{\%} of cases, leaving ample opportunity for future improvement.",
}
```


