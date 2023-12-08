import os

from transformers import AutoTokenizer, MBartForConditionalGeneration


###
# やさしい日本語変換用
def translate_one_sentence(source_sentence):
    # トークナイズ
    inputs = tokenizer(source_sentence, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    # モデルで翻訳を実行
    outputs = model.generate(**inputs)
    # トークンをテキストにデコード
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("simplify:", translated_sentence)

    return translated_sentence


model_name = "ku-nlp/bart-large-japanese"
# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)

# モデルのパスを指定
model_path = "models/checkpoint-12500"
model_path = os.path.join(os.path.dirname(__file__), model_path)
# モデルのロード
model = MBartForConditionalGeneration.from_pretrained(model_path)

# translate_one_sentence("森川さんはしょっちゅう何か不平を言っている。")
###
