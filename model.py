import re

import ginza
import spacy
import torch
from transformers import BertModel, BertTokenizer

re_kanji = re.compile(r'^[\u4E00-\u9FD0]+$')

# GPU on M1 Mac
device = torch.device("mps")
tokenizer = BertTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
c = spacy.load('ja_ginza')


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            return_dict=False,
        )
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)  # 二値分類のため，出力層は1次元？
        self.l4 = torch.nn.Dropout(0.3)
        self.l5 = torch.nn.Linear(768, 1)  # 二値分類のため，出力層は1次元？

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        output_2 = self.l2(output_1)
        output_lf = self.l3(output_2)

        output_3 = self.l2(output_1)
        output_p = self.l3(output_3)

        return output_lf, output_p


def load_best(best_model_path, model):
    """
    best_model_path: path to best model
    model: model that we want to load checkpoint parameters into
    """
    # load check point
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def insertion(text):
    # cabocha で解析
    # tree = c.parse(text)

    linefeed_text = ''

    # ginza で解析
    doc = c(text)
    for sent in doc.sents:

        length = 0
        old_chunk_text = ''
        for span in ginza.bunsetu_spans(sent):
            chunk_text = span.text
            print(chunk_text)  # debug

            length += len(chunk_text)

            test_text = old_chunk_text + "[SEP]" + chunk_text
            with torch.no_grad():
                inputs = tokenizer.encode_plus(
                    test_text,
                    None,
                    add_special_tokens=True,
                    max_length=50,
                    padding='max_length',
                    return_token_type_ids=True,
                    truncation=True
                )
                ids = inputs['input_ids']
                mask = inputs['attention_mask']
                token_type_ids = inputs["token_type_ids"]
                ids = torch.tensor([ids], dtype=torch.long).to(device, dtype=torch.long)
                mask = torch.tensor([mask], dtype=torch.long).to(device, dtype=torch.long)
                token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device, dtype=torch.long)
                features = torch.tensor([features], dtype=torch.float).to(device, dtype=torch.float)

                # print(ids)  # 読み込んだテキストを id 化したものの確認, debug

                output_lf, output_p = model(ids, mask, token_type_ids)

            pred_lf = torch.sigmoid(output_lf).cpu().detach().numpy().tolist()  # 改行
            pred_p = torch.sigmoid(output_p).cpu().detach().numpy().tolist()  # 読点

            print(pred_lf[0][0], pred_p[0][0])

            if pred_p[0][0] > 0.5:
                print("読点")
                linefeed_text += "、"  # 直前の文節境界に改行を付与
                length += 1

            if pred_lf[0][0] > 0.5:
                print("モデルによる改行")
                # linefeed_text += "<br>"  # 直前の文節境界に改行を付与
                linefeed_text += "\n"  # for gradio
                length = len(chunk_text)
            elif length > 20:  # 最終文節を繋いだときに20文字を超えていたら
                print("文字数による改行")
                # linefeed_text += "<br>"  # 直前の文節境界に改行を付与
                linefeed_text += "\n"  # for gradio
                length = len(chunk_text)

            linefeed_text += chunk_text

            old_chunk_text = chunk_text
        linefeed_text += "。\n"

    print(linefeed_text)
    return linefeed_text


model = BERTClass()
model.to(device)
best_model = './insertion_lf-p_model.pt'
model = load_best(best_model, model)