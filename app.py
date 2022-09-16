from datetime import timedelta

import CaboCha
import torch
from flask import Flask  # Flaskと、HTMLをレンダリングするrender_templateをインポート
from flask import Markup, render_template, request, session
from transformers import BertTokenizer

from model import BERTClass, load_best

# GPU on M1 Mac
device = torch.device("mps")
tokenizer = BertTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model = BERTClass()
model.to(device)
best_model = './insertion_model.pt'
model = load_best(best_model, model)

app = Flask(__name__)  # Flask の起動

# session の利用
app.secret_key = 'secret'
# session は 3 分で破棄
app.permanent_session_lifetime = timedelta(minutes=3)

# cabocha の解析記
c = CaboCha.Parser()


def lf_and_punc_insertion(text):
    # cabocha で解析
    tree = c.parse(text)

    length = 0
    linefeed_text = ''
    old_chunk_text = ''
    for i in range(tree.chunk_size()):  # 文節数のループ
        chunk = tree.chunk(i)  # chunk=文節．i 番目の文節

        chunk_text = ''
        # 文節内の各形態素解析結果を表示
        for ix in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            token = tree.token(ix)
            # token.normalized_surface  # 表層
            # token.feature  # 形態素情報

            chunk_text += token.normalized_surface  # 形態素ごとに繋いでいく

        print(chunk_text)

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

            # print(ids)  # 読み込んだテキストを id 化したものの確認

            output = model(ids, mask, token_type_ids)

        pred = torch.sigmoid(output).cpu().detach().numpy().tolist()
        print(pred[0][0])
        if pred[0][0] > 0.5:
            print("モデルによる改行")
            linefeed_text += "<br>"  # 直前の文節境界に改行を付与
            length = len(chunk_text)
        elif length > 20:  # 最終文節を繋いだときに20文字を超えていたら
            print("文字数による改行")
            linefeed_text += "<br>"  # 直前の文節境界に改行を付与
            length = len(chunk_text)

        linefeed_text += chunk_text

        old_chunk_text = chunk_text

    print(linefeed_text)

    return linefeed_text


@app.route('/')  # localhost:50000/を起動した際に実行される
def index():
    return render_template('index.html')  # index.htmlをレンダリングする


@app.route("/main")
def show_iframe():
    return render_template(
        "main.html",
        bgcolor="#00ff00",
        bottom=0,
        textAlign="left",
        stwidth=6,
        fontsize=25,
        fontweight=900,
        stylecolor="#ffffff",
        stcolor="#000000",
        linefeed_text="[ここに結果表示（音声認識）]"
    )


@app.route('/result', methods=['GET', 'POST'])
def result():
    # session に保存してあるテキストを読み込み
    text = session.get('text', '')

    text_split = text.split('<br>')  # 改行<br>で分割
    pre = "<br>".join(text_split[:-1]) + "<br>"  # 最終の改行までの文字列（確定済み）
    now = text_split[-1]  # 最終の改行以降の文字列

    # index.htmlのinputタグ内にあるname属性recog_textを取得し、nowに結合
    now += request.form.get('recog_text')

    # 改行挿入
    linefeed_text = pre + lf_and_punc_insertion(now)

    # 字幕の表示は5行（6行以上になったら5行にカット）
    if linefeed_text.count("<br>") >= 5:
        linefeed_text = "<br>".join(linefeed_text.split("<br>")[-5:])
    print(linefeed_text)

    # 改行挿入結果を session に保存
    session['text'] = linefeed_text

    # 表示
    return render_template(
        "main.html",
        bgcolor="#00ff00",
        bottom=0,
        textAlign="left",
        stwidth=6,
        fontsize=25,
        fontweight=900,
        stylecolor="#ffffff",
        stcolor="#000000",
        linefeed_text=Markup(linefeed_text)
    )


if __name__ == '__main__':
    app.run(debug=True)
