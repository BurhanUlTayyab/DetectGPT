"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

from torch import equal
from model import GPT2PPLV2 as GPT2PPL
from fastapi import FastAPI, Form
from fastapi import Request
import gradio as gr
import uvicorn
from database import DB
from HTML_MD_Components import noticeBoardMarkDown, bannerHTML, emailHTML, discordHTML

CUSTOM_PATH = "/"

app = FastAPI()

# initialize the model
model = GPT2PPL()
database  = DB()

@app.post("/postdb")
def uploadDataBase(email: str = Form(), request: Request = None):
    database.set(request.client.host, email)
    return "Email Sent"

@app.get("/infer")
def infer(sentence: str):
    return model(sentence=sentence)

def inference(*args):
    return model(*args)

with gr.Blocks(title="SG-GPTZero", css="#discord {text-align: center} #submit {background-color: #FF8C00} #advertisment {text-align: center;} #email {height:120%; background-color: LightSeaGreen} #blank {margin:150px} #code_feedback { margin-left:-0.3em;color:gray;text-align: center;margin-bottom:-100%;padding-bottom:-100%}") as io:
    with gr.Row():
         gr.HTML(bannerHTML, visible=True)
    with gr.Row():
        with gr.Column(scale=0.1):
            pass
        with gr.Column(scale=0.8):
            gr.Markdown('<h1 style="text-align: center;">SG-GPTZero (<a style="text-decoration:none" href="https://github.com/BurhanUlTayyab/GPTZero">Code V1</a>/<a style="text-decoration:none" href="https://github.com/BurhanUlTayyab/DetectGPT">Code V1.1</a>)</h1>')
        with gr.Column(scale=0.1, elem_id="discord"):
            gr.HTML(discordHTML, visible=True)
    with gr.Row():
        gr.Markdown("Use SG-GPTZero to determine if the text is written by AI or Human.")
    with gr.Tab("V1.1 (Detect GPT)"):
        with gr.Row(elem_id="row1"):
            with gr.Column(scale=1):
                InputTextBox = gr.Textbox(lines=7, placeholder="Please Insert your text(s) here", label="Texts")
                sumbit_btn = gr.Button("Submit", elem_id="submit")
            with gr.Column(scale=1):
                OutputLabels = gr.JSON(label="Output")
                OutputTextBox = gr.Textbox(show_label=False)
                with gr.Accordion("Details", open=False):
                    with gr.Box():
                        OutputHighlightedText = gr.HTML(show_label=False)

        sumbit_btn.click(lambda x: inference(x, 512, "v1.1"), inputs=[InputTextBox], outputs=[OutputLabels, OutputTextBox, OutputHighlightedText], api_name="infer")
    with gr.Tab("V1 (GPT-Zero)"):
        with gr.Row(elem_id="row1"):
            with gr.Column(scale=1):
                InputTextBox_v1 = gr.Textbox(lines=7, placeholder="Please Insert your text(s) here", label="Texts")
                sumbit_btn_v1 = gr.Button("Submit", elem_id="submit")
            with gr.Column(scale=1):
                OutputLabels_v1 = gr.JSON(label="Output")
                OutputTextBox_v1 = gr.Textbox(show_label=False)
        sumbit_btn_v1.click(lambda x: inference(x, "v1"), inputs=InputTextBox_v1, outputs=[OutputLabels_v1, OutputTextBox_v1], api_name="infer")
    with gr.Tab("V2 (Coming Soon)"):
        gr.Markdown('<p style="text-align:center"> Something interest is coming...</p>')
    with gr.Row():
        with gr.Box():
            gr.Markdown(noticeBoardMarkDown(), visible=True)
    with gr.Row():
        gr.Markdown('# <span style="color:#006400">Register</span> here for updates.')
    with gr.Row():
        with gr.Column(scale=0.5):
            emailTextBox = gr.HTML(emailHTML)
        with gr.Column(scale=0.5):
            pass
 
    with gr.Row():
        gr.Markdown('<span style="color:red">Do you want to train Computer vision models faster and with less data?. Visit [Ailiverse](https://ailiverse.com)</span> <br> <span style="color:gray"><p>Powered by Ailiverse </p></span>', elem_id="advertisment")
    with gr.Row():
        gr.Markdown('For <a style="text-decoration:none;color:gray" href="mailto:gptzero@ailiverse.com" target="_blank">feedback</a>, contact us at gptzero@ailiverse.com', elem_id="code_feedback")

app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)
