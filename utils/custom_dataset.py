import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # tokenize the text data, prepare the model input
    def __init__(self, dataframe, tokenizer, max_len, label = True):
        self.len = len(dataframe)
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.label = label

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        text = str(self.data.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        if self.label:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.data.label[index], dtype=torch.long)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            }