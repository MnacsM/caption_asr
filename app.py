from datetime import timedelta

import CaboCha
from flask import Flask  # Flaskと、HTMLをレンダリングするrender_templateをインポート
from flask import Markup, render_template, request, session

app = Flask(__name__)  # Flask の起動
app.secret_key = 'secret'
# session は 3 分で破棄
app.permanent_session_lifetime = timedelta(minutes=3)

c = CaboCha.Parser()


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
    # index.htmlのinputタグ内にあるname属性recog_textを取得し、textに結合
    text += request.form.get('recog_text')
    # session に保存
    session['text'] = text

    # cabocha で解析
    tree = c.parse(text)

    length = 0
    linefeed_text = ''
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
        if length > 20:  # 最終文節を繋いだときに20文字を超えていたら
            linefeed_text += "<br>"  # 直前の文節境界に改行を付与
            length = len(chunk_text)

        linefeed_text += chunk_text
    print(linefeed_text)

    # 字幕の表示は5行（6行以上になったら5行にカット）
    if linefeed_text.count("<br") >= 5:
        linefeed_text = "<br>".join(linefeed_text.split("<br>")[1:])
    print(linefeed_text)

    # もしPOSTメソッドならresult.htmlに値textと一緒に飛ばす
    if request.method == 'POST':
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

    # POSTメソッド以外なら、index.htmlに飛ばす
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
