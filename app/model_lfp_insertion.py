import os
import re
from argparse import ArgumentParser

import ginza
import spacy
import torch
from transformers import BertModel, BertTokenizer

# GPU on M1 Mac
# parser = ArgumentParser()
# parser.add_argument('--device', type=str, default='mps',
#                     choices=['cpu', 'mps', 'cuda'])
# args = parser.parse_args()
# device = torch.device(args.device)

re_kanji = re.compile(r'^[\u4E00-\u9FD0]+$')


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            return_dict=False,
        )
        # 改行
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
        # 読点
        self.l4 = torch.nn.Dropout(0.3)
        self.l5 = torch.nn.Linear(768 + 0, 1)  # 768 は bert の次元数．そこに独自の素性の数を足す

    def forward(self, ids, mask, token_type_ids, features):
        _, output_1 = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        output_2 = self.l2(output_1)
        output_lf = self.l3(output_2)

        output_3 = self.l4(output_1)
        # output_3 = torch.cat([output_3, features], dim=1)  # embed と素性を連結
        output_p = self.l5(output_3)

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


def lf_p_insertion(text):
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

            # features
            if old_chunk_text != '':
                before_status = re_kanji.fullmatch(old_chunk_text[-1])
                after_status = re_kanji.fullmatch(chunk_text[0])
                # 漢字が連続
                if before_status and after_status:
                    features = [1]
                else:
                    features = [-1]
            else:
                features = [-1]

            if old_chunk_text != '':
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

                    # output_lf, output_p = model(ids, mask, token_type_ids)
                    output_lf, output_p = model(ids, mask, token_type_ids, features)

                pred_lf = torch.sigmoid(output_lf).cpu().detach().numpy().tolist()  # 改行
                pred_p = torch.sigmoid(output_p).cpu().detach().numpy().tolist()  # 読点

                print(pred_lf[0][0], pred_p[0][0])

                # if pred_p[0][0] > 0.5:
                #     print("読点")
                #     linefeed_text += "、"  # 直前の文節境界に改行を付与
                #     length += 1

                if pred_lf[0][0] > 0.5:
                    print("モデルによる改行")
                    linefeed_text += "<br>"  # 直前の文節境界に改行を付与
                    length = len(chunk_text)
                elif length > 20:  # 最終文節を繋いだときに20文字を超えていたら
                    print("文字数による改行")
                    linefeed_text += "<br>"  # 直前の文節境界に改行を付与
                    length = len(chunk_text)

            linefeed_text += chunk_text

            old_chunk_text = chunk_text

        if sent.text.endswith('。'):
            linefeed_text += "<br>"
        else:
            linefeed_text += "。<br>"  # for gradio

    print("insertion:", linefeed_text)
    return linefeed_text


device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
c = spacy.load('ja_ginza')

model = BERTClass()
model.to(device)

best_model = os.path.join(os.path.dirname(__file__), 'models/insertion_lf-p_model_cpu.pt')
model = load_best(best_model, model)
print("load best model")

# docker用にcpuに変更したものを保存
# checkpoint = {
#     "epoch": None,
#     "valid_loss_min": None,
#     "state_dict": model.state_dict(),
#     "optimizer": None,
#     }
# torch.save(checkpoint, os.path.join(os.path.dirname(__file__), 'models/insertion_lf-p_model_cpu.pt'))
# print("done")
