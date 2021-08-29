import re
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Data():
    def __init__(self, hparams):
        self.hparams = hparams
        self.file_path = hparams.data_path
        self.max_length = hparams.max_length

        self.load_datasets()

    def load_datasets(self):
        raise NotImplementedError
    

class LabeledDocuments(Data):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def load_datasets(self):
        self.X_train, self.Y_train = self.read_data(f"{self.file_path}/train.csv", max_length=self.max_length)
        self.X_val, self.Y_val = self.read_data(f"{self.file_path}/val.csv", max_length=self.max_length)
        self.X_test, self.Y_test = self.read_data(f"{self.file_path}/test.csv", max_length=self.max_length)

        print(f"train size: {len(self.X_train)}, validation size: {len(self.X_val)}, test size: {len(self.X_test)}")
    
    def read_data(self, path, max_length):
        def label_fn(x):
            return x - 1
        
        rows = pd.read_csv(
            path,
            sep=",",
            error_bad_lines=False,
            header=None,
            skiprows=None,
            quoting=0,
            keep_default_na=False,
            encoding="utf-8",
        )

        label_fn = label_fn if label_fn is not None else (lambda x: x)
        labels = rows[0].apply(lambda x: label_fn(x))
        sentences = rows[1]
        sentences = sentences.apply(lambda x: clean_tokenize_truncate(x, max_length))
        return sentences.tolist(), labels.tolist()
    
    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        train_dataset = My_Dataset(self.X_train, self.Y_train, self.max_length)
        database_dataset = My_Dataset(self.X_train, self.Y_train, self.max_length)
        val_dataset = My_Dataset(self.X_val, self.Y_val, self.max_length)
        test_dataset = My_Dataset(self.X_test, self.Y_test, self.max_length)

        # DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    shuffle=shuffle_train, num_workers=num_workers)
        database_loader = DataLoader(dataset=database_dataset, batch_size=512,
                                    shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=512,
                                    shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=512,
                                    shuffle=False, num_workers=num_workers) if get_test else None
        return train_loader, database_loader, val_loader, test_loader, 

class My_Dataset(Dataset):
    def __init__(self, data, labels, max_length):
        self.data = data
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("model/bert-base-uncased")
    
    def __getitem__(self, index):
        sentence = " ".join(self.data[index]).lower()
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length+1, return_tensors="pt")
        return inputs, self.labels[index]
    
    def __len__(self):
        return len(self.data)

def clean_string(string):
    """
    Tokenization/string cleaning for yelp data set
    Based on https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\"\"", ' " ', string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def clean_tokenize_truncate(x, max_length):
    x = clean_string(x)
    x = x.split(" ")
    x = x[:max_length]
    return x
