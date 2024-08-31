
import gradio as gr

from datetime import datetime,timedelta

def app():
    with gr.Blocks(title="Tokyo Disney Resort Route Optimzer") as demo:
        gr.Markdown("# Tokyo Disney Resort Route Optimzer")

        with gr.Tabs():
            # home
            with gr.TabItem("Create Plan"):
                with gr.Row():
                    # setting of date
                    # setting of time
                with gr.Accordion(label="Setting", open=False):
                    # tasks
                    # 隠しパラメータの設定
                    with gr.Row():
                    with gr.Row():
                submit_btn = gr.Button("execute", variant="primary")
            ## result
            with gr.TabItem("Your Plan"):
                outputbox = gr.Markdown(label="output", elem_id="outputbox")

            ## 説明書
            with gr.TabItem("README"):
                gr.Markdown(read_markdown(str(Path(__file__).resolve().parents[1] / "doc/app.md")))

            # Action
            submit_btn.click(
                fn=generate_agenda,
                inputs=[
                ],
                outputs=[outputbox, output_download, output_zip],
            )
    return demo