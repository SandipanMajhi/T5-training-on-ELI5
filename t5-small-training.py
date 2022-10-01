import nltk
nltk.download('punkt')
import numpy as np
import transformers
# from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


class T5_Small_Custom:
    '''
        This class is a set of functions and modules that trains a QA dataset - ELI5
    '''

    def __init__(self, model_checkpoint, dataset_name, metric_name, 
                    prefixes, max_input_length = 128, max_target_length = 512 ):

        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.raw_datasets = load_dataset(dataset_name)
        self.metric = load_metric(metric_name)  
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.prefixes = prefixes
        self.input_len = max_input_length
        self.target_len = max_target_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    def preprocess_data(self, examples):
        '''
            Here, we concatenate the prefixes before the question and support prefix before the content
        '''
        inputs = [self.prefixes[0] + " " + doc[0] + " " + self.prefixes[1] + " " + doc[1] for doc in zip(examples["title"], examples['selftext'])]
        model_inputs = self.tokenizer(inputs, max_length = self.input_len, truncation = True)
        with self.tokenizer.as_target_tokenizer():
            texts = [x['text'][0] for x in examples['answers']]
            labels = self.tokenizer(["[SEP]".join(p['text']) for p in examples['answers']], max_length = self.target_len, truncation = True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, Epochs, train_name, validation_name, batch_size, lr, strategy, weight_decay,
                 save_limit, fp16, push_to_hub = True):

        self.tokenized_dataset = self.raw_datasets.map(self.preprocess_data, batched = True)
        self.model_name = self.model_checkpoint.split("/")[-1]

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

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.trainer = Seq2SeqTrainer(
            self.model,
            self.args,
            train_dataset=self.tokenized_dataset[train_name],
            eval_dataset=self.tokenized_dataset[validation_name],
            data_collator= self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()
        self.trainer.push_to_hub()

    
    def compute_accuracy(self, split_name):
        return self.trainer.evaluate(self.tokenized_dataset[split_name])


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}  



def main():

    model_checkpoint = "t5-small"
    dataset_name = "eli5"
    metric_name = "rouge"
    prefixes = ["question:","context:"]

    t5_custom = T5_Small_Custom(model_checkpoint= model_checkpoint, dataset_name= dataset_name, metric_name= metric_name, prefixes=prefixes)
    t5_custom.train(Epochs = 1, train_name = "train_eli5", validation_name = "validation_eli5", 
                    batch_size = 16, lr = 2e-5, strategy = "epoch" , weight_decay = 0.01, save_limit = 3, fp16 = True)

    validation_metric = t5_custom.compute_accuracy("validation_eli5")
    test_metrics = t5_custom.compute_accuracy("test_eli5")

    print("Validation Metrics ------------------------ ")
    print(validation_metric)

    print("Test Metric ------------------------")
    print(test_metrics)



if __name__ == "__main__" :
    main()