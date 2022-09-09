from datetime import timedelta

import ipadic
import MeCab
from flask import Flask  # Flaskと、HTMLをレンダリングするrender_templateをインポート
from flask import Markup, render_template, request, session

app = Flask(__name__)  # Flask の起動
app.secret_key = 'secret'
app.permanent_session_lifetime = timedelta(minutes=3)


t = MeCab.Tagger(f'-O wakati {ipadic.MECAB_ARGS}')


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
        linefeed_text=''
    )


@app.route('/result', methods=['GET', 'POST'])
def result():
    # index.htmlのinputタグ内にあるname属性itemを取得し、textに格納した
    text = session.get('text', '')
    text += request.form.get('item')
    session['text'] = text
    text_parse = t.parse(text)

    length = 0
    linefeed_text = ''
    for token in text_parse.split(' '):
        length += len(token)
        if length > 20:
            linefeed_text += "<br>"
            length -= 20

        linefeed_text += token

    # もしPOSTメソッドならresult.htmlに値textと一緒に飛ばす
    if request.method == 'POST':
        print(text)
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
