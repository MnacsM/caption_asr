from datetime import timedelta

from flask import Flask  # Flaskと、HTMLをレンダリングするrender_templateをインポート
from flask import Markup, render_template, request, session

from model import insertion

app = Flask(__name__)  # Flask の起動

# session の利用
app.secret_key = 'secret'
# session は 3 分で破棄
app.permanent_session_lifetime = timedelta(minutes=3)


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
    linefeed_text = pre + insertion(now)

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
