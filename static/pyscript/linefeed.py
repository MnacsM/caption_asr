from js import document
from pyodide import create_proxy

asr_frame = document.getElementById('asr_frame').contentWindow.document.getElementById('speech_text-imb')

async def asrchange(event):
    asr_now_text = asr_frame.innerHTML
    if "けれども" in asr_now_text:
        asr_now_text = asr_now_text.replace("けれども", "けれども（改行）")
    pyscript.write("pyscriptsample", asr_now_text)

asr_frame.addEventListener("DOMNodeInserted", create_proxy(asrchange))
