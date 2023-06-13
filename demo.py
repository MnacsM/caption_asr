import gradio as gr

from app.model import insertion

examples = [
    "例えば環境の問題あるいは人口の問題エイズの問題などなど地球規模の問題たくさん生じておりますが残念ながらこれらの問題は二十一世紀にも継続しあるいは悲観的な見方をすればさらに悪くなるという風に思われます",
    "今日はとてもいい天気でしたが少し暑かったので水分が多く必要になりました",
]


def greet(text):
    return insertion(text)


demo = gr.Interface(
    fn=greet,
    inputs=gr.inputs.Textbox(lines=5, label="input text"),
    outputs=gr.outputs.Textbox(label="inserted"),
    examples=examples
)

demo.launch()
