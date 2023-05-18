import gradio as gr
import utils
from api import AutoAPI, get_openai_api_key
import os, shutil
import json

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(FILE_DIR), "auto_gpt_workspace")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

CSS = """
#chatbot {font-family: monospace;}
#files .generating {display: none;}
#files .min {min-height: 0px;}
"""

with gr.Blocks(css=CSS) as app:
    with gr.Column() as setup_pane:
        gr.Markdown(f"""# Auto-GPT
        """)
        with gr.Row():
            open_ai_key = gr.Textbox(
                value=get_openai_api_key(),
                label="OpenAI API Key",
                type="password",
            )
        gr.Markdown(
            "Fill the values below, then click 'Start'. There are example values you can load at the bottom of this page."
        )
        with gr.Row():
            ai_name = gr.Textbox(label="AI Name", placeholder="e.g. Entrepreneur-GPT")
            ai_role = gr.Textbox(
                label="AI Role",
                placeholder="e.g. an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.",
            )
        top_5_goals = gr.Dataframe(
            row_count=(5, "fixed"),
            col_count=(1, "fixed"),
            headers=["AI Goals - Enter up to 5"],
            type="array"
        )
        start_btn = gr.Button("Start", variant="primary")
        with open(os.path.join(FILE_DIR, "examples.json"), "r") as f:
            example_values = json.load(f)
        gr.Examples(
            example_values,
            [ai_name, ai_role, top_5_goals],
        )
    with gr.Column(visible=False) as main_pane:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(elem_id="chatbot")
                with gr.Row():
                    yes_btn = gr.Button("Yes", variant="primary", interactive=False)
                    consecutive_yes = gr.Slider(
                        1, 10, 1, step=1, label="Consecutive Yes", interactive=False
                    )
                custom_response = gr.Textbox(
                    label="Custom Response",
                    placeholder="Press 'Enter' to Submit.",
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.HTML(
                    lambda: f"""
                        Generated Files
                        <pre><code style='overflow-x: auto'>{utils.format_directory(OUTPUT_DIR)}</pre></code>
                """, every=3, elem_id="files"
                )
                download_btn = gr.Button("Download All Files")

    chat_history = gr.State([[None, None]])
    api = gr.State(None)

    def start(open_ai_key, ai_name, ai_role, top_5_goals):
        auto_api = AutoAPI(open_ai_key, ai_name, ai_role, top_5_goals)
        return gr.Column.update(visible=False), gr.Column.update(visible=True), auto_api

    def bot_response(chat, api):
        messages = []
        for message in api.get_chatbot_response():
            messages.append(message)
            chat[-1][1] = "\n".join(messages) + "..."
            yield chat
        chat[-1][1] = "\n".join(messages)
        yield chat

    def send_message(count, chat, api, message="Y"):
        if message != "Y":
            count = 1
        for i in range(count):
            chat.append([message, None])
            yield chat, count - i
            api.send_message(message)
            for updated_chat in bot_response(chat, api):
                yield updated_chat, count - i

    def activate_inputs():
        return {
            yes_btn: gr.Button.update(interactive=True),
            consecutive_yes: gr.Slider.update(interactive=True),
            custom_response: gr.Textbox.update(interactive=True),
        }

    def deactivate_inputs():
        return {
            yes_btn: gr.Button.update(interactive=False),
            consecutive_yes: gr.Slider.update(interactive=False),
            custom_response: gr.Textbox.update(interactive=False),
        }

    start_btn.click(
        start,
        [open_ai_key, ai_name, ai_role, top_5_goals],
        [setup_pane, main_pane, api],
    ).then(bot_response, [chat_history, api], chatbot).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )

    yes_btn.click(
        deactivate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    ).then(
        send_message, [consecutive_yes, chat_history, api], [chatbot, consecutive_yes]
    ).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )
    custom_response.submit(
        deactivate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    ).then(
        send_message,
        [consecutive_yes, chat_history, api, custom_response],
        [chatbot, consecutive_yes],
    ).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )

    def download_all_files():
        shutil.make_archive("outputs", "zip", OUTPUT_DIR)

    download_btn.click(download_all_files).then(None, _js=utils.DOWNLOAD_OUTPUTS_JS)

app.queue(concurrency_count=20).launch(file_directories=[OUTPUT_DIR])
