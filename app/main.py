import os
from datetime import timedelta

from flask import Flask  # Flaskと、HTMLをレンダリングするrender_templateをインポート
from flask import render_template, request, session
from markupsafe import Markup
from model_lfp_insertion import lf_p_insertion
from model_text_simplify import translate_one_sentence

app = Flask(__name__)  # Flask の起動

# session の利用
app.secret_key = 'secret'
# session は 3 分で破棄
app.permanent_session_lifetime = timedelta(minutes=3)

MAX_LINES = 10


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

    # index.htmlのinputタグ内にあるname属性recog_textを取得
    recog_text = request.form.get('recog_text')
    print("recog_text:", recog_text)

    # やさしい日本語変換
    recog_text = translate_one_sentence(recog_text)

    # recog_text を nowに結合
    now += recog_text
    # 改行挿入
    linefeed_text = pre + lf_p_insertion(now)

    # 字幕の表示は MAX_LINES 行（MAX_LINES+1行以上になったら MAX_LINES 行にカット）
    if linefeed_text.count("<br>") >= MAX_LINES:
        linefeed_text = "<br>".join(linefeed_text.split("<br>")[-MAX_LINES:])

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
    SERVER_CRT = os.path.join(os.path.dirname(__file__), 'openssl/server.crt')
    SERVER_KEY = os.path.join(os.path.dirname(__file__), 'openssl/server.key')

    app.run(
        host='0.0.0.0',
        port=334,
        ssl_context=(SERVER_CRT, SERVER_KEY),
        threaded=True, debug=True
    )
