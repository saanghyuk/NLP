import torch
from torch.utils.data import Dataset


class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=True):
        # tokenizer : hugging face tokenizer object
        self.tokenizer = tokenizer
        # set maximum length
        self.max_length = max_length
        self.with_text = with_text
    # 'samples' will be the list of dictionary return of TextClassificationDataset.
    # [{} , {}, {}]
    #  return {
    #         'text': text,
    #         'label': label,
    #     }

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        # texts and labels will be the list of text and label from each mini-batch

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        # |encoding| = (N, length, 1)
        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }

        # with_text will be true if the original text is needed
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):
    # Input List of Dataset : dataset & Label
    # texts, labels are 'list' format
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    # this means the number of sample of dataset
    def __len__(self):
        return len(self.texts)

    # if the mini-batch size is 128,
    # Dataloader will cal __getitem__ function 128 times.
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }
