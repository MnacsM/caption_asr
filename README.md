# ブラウザで字幕表示

音声認識の結果に対して改行と読点の挿入を行い，字幕として表示する．
1行の文字数は20文字以下，字幕は5行以内の表示とする．

字幕の表示部分は西村良太先生の「音声認識字幕ちゃん」を利用させていただく．
- http://www.sayonari.com/trans_asr/index_asr.html
- https://github.com/sayonari/jimakuChan

## Algorithm

- 音声認識の最終結果を受け取る
  - 音声認識の途中結果は，「音声認識字幕ちゃん」に従い <<結果>> で表示
- 音声認識結果に文節まとめ上げを行い，**最終の文節境界以外** に対して順に挿入判定を行う
- 挿入した結果を出力する
- 挿入結果は session に保存する

## Usage

```shell
$ python main.py
```

## require

- MeCab, CaboCha
- Flask
