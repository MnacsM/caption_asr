import torch
from transformers import BertModel


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            return_dict=False,
        )
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)  # 二値分類のため，出力層は1次元？

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def load_best(best_model_path, model):
    """
    best_model_path: path to best model
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model
