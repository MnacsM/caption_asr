import os
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

from flask import Flask, redirect, render_template, request, session, url_for
from markupsafe import Markup
from model import Insertion, Simplify

app = Flask(__name__)  # Flask の起動

app.secret_key = 'secret'  # session の利用
app.permanent_session_lifetime = timedelta(minutes=3)  # session は 3 分で破棄

MAX_LINES = 10

insertion = Insertion()
simplify = Simplify()


@app.route('/')  # https://127.0.0.1:334/を起動した際に実行される
def index():
    return render_template('index.html')  # index.htmlをレンダリングする


@app.route("/main")
def show_iframe():
    config = load_config()

    text = session.get('text', '')
    if text == '':
        text = "[ここに結果表示（音声認識）]"

    return render_template(
        "main.html",
        **config,
        linefeed_text=Markup(text)
    )


@app.route('/result', methods=['GET', 'POST'])
def result():
    # session から text を取得
    text = session.get('text', '')

    text_split = text.split('<br>')  # 改行<br>で分割
    pre = "<br>".join(text_split[:-1]) + "<br>"  # 最終の改行までの文字列（確定済み）
    now = text_split[-1]  # 最終の改行以降の文字列

    # index.htmlのinputタグ内にあるname属性recog_textを取得
    recog_text = request.form.get('recog_text')
    print("recog_text:", recog_text)

    # やさしい日本語変換
    recog_text = simplify.simplify(recog_text)

    # recog_text を nowに結合
    now += recog_text
    # 改行挿入
    linefeed_text = pre + insertion.insertion(now)

    # 字幕の表示は MAX_LINES 行（MAX_LINES+1行以上になったら MAX_LINES 行にカット）
    if linefeed_text.count("<br>") >= MAX_LINES:
        linefeed_text = "<br>".join(linefeed_text.split("<br>")[-MAX_LINES:])

    # 改行挿入結果を session に保存
    session['text'] = linefeed_text

    # session から config を取得
    config = load_config()

    # 表示
    return render_template(
        "main.html",
        **config,
        linefeed_text=Markup(linefeed_text)
    )


@app.route('/config', methods=['GET', 'POST'])
def set_config():
    url = request.form.get('url')

    # URLを解析してクエリパラメータを取得
    url = url.replace("#", "%23")
    config = urlparse(url)
    config = parse_qs(config.query)

    # session に保存
    session['config'] = config

    return redirect(url_for('index'))


def load_config():
    # session から config を取得
    config = session.get('config', '')
    if len(config) == 0:
        config = {
            'textAlign': 'left',
            'v_align': 'bottom',
            'recog': 'ja',
            'bgcolor': '#00ff00',
            'size1': 25,
            'weight1': 900,
            'color1': '#ffffff',
            'st_color1': '#000000',
            'st_width1': 6,
            'speech_text_font': 'M PLUS Rounded 1c',
            'short_pause': 750
        }
    else:
        config = {
            'textAlign': config['textAlign'][0],
            'v_align': config['v_align'][0],
            'recog': config['recog'][0],
            'bgcolor': config['bgcolor'][0],
            'size1': int(config['size1'][0]),
            'weight1': int(config['weight1'][0]),
            'color1': config['color1'][0],
            'st_color1': config['st_color1'][0],
            'st_width1': int(config['st_width1'][0]),
            'speech_text_font': config['speech_text_font'][0],
            'short_pause': int(config['short_pause'][0])
        }

    return config


if __name__ == '__main__':
    SERVER_CRT = os.path.join(os.path.dirname(__file__), 'openssl/server.crt')
    SERVER_KEY = os.path.join(os.path.dirname(__file__), 'openssl/server.key')

    app.run(
        host='0.0.0.0',
        port=334,
        ssl_context=(SERVER_CRT, SERVER_KEY),
        threaded=True, debug=True
    )
