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
the supporting document.

Run the python file on GPU. In the file, edit the following code block,

'''
  t5_custom = T5_Small_Custom(model_checkpoint= model_checkpoint, dataset_name= dataset_name, metric_name= metric_name, prefixes=prefixes)
  t5_custom.train(Epochs = 1, train_name = "train_eli5", validation_name = "validation_eli5", 
                      batch_size = 16, lr = 2e-5, strategy = "epoch" , weight_decay = 0.01, save_limit = 3, fp16 = True)
'''
