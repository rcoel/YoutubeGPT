import gradio as gr
from model import YoutubeLink

link = YoutubeLink()

examples = [
    'https://www.youtube.com/watch?v=JN3KPFbWCy8&t=237s',
    'https://www.youtube.com/watch?v=L_Guz73e6fw&t=2950s'
]

def response(message, history):
    if message == '' and link.video_url == None:
        return "Enter a youtube url"
    elif message == '':
        return "If you want to know anything in particular, feel free to type it"
    elif "https://" in message:
        link.db = None
        link.video_url = message
        return link.get_summary()
    else:
        if link.video_url == None:
            return "Enter a youtube url!!"
        else:
            return link.answer_query(message)


chat_bot = gr.ChatInterface(fn = response, retry_btn=False, stop_btn=False, clear_btn= False, undo_btn= False, title= "YoutubeGPT", examples= examples)



with gr.Blocks() as details:
    gr.Markdown(
    """

# YoutubeGPT: A Video Summarization and Question Answering Chatbot

YoutubeGPT is an innovative project that leverages advanced natural language processing techniques to enhance the video consumption experience. The core functionalities include video summarization and question-answering capabilities, making it a versatile tool for users seeking quick insights from diverse content on YouTube.


## Working

The project employs the RAG (Retrieval-Augmented Generation) approach, utilizing Zephyr-7B-Beta as a powerful Language Model (LLM). Langchain serves as a crucial LLM wrapper, enhancing the overall efficiency of the language processing pipeline. Additionally, Chroma acts as a sophisticated vector store, facilitating seamless storage and retrieval of information.

"""
)
    gr.Image('flowchart.png', width= 500, show_download_button= False, show_label= False)

    gr.Markdown(
        """
## Comparision between text-generation models

We compared the text-generation capabilities of two major models: <u>**Llama2-Chat**</u> and <u>**Zephyr-7b-Beta**</u>
The result of the comparision was found as:
| Model             | MT-Bench  | MMLU   | HellaSwag | AlpacaEval |
| -------------     | ---       | --------| ------- | --------- |
| LLaMA 2 7B Chat  | 6.86      |  44.4%  | 77.1%   | 71.37%|
| Zephyr 7B beta    | **7.34**  | **60.1**| **81.3%** | **90.60%** |


We can see that Zephyr-7B beta easily beats LLama 2 7B chat.

Below are the graphs comparing the performances of Zephyr 7B(Mistral) with LLaMA series

"""
    )
    gr.Image('graph.png', width= 800, show_download_button= False, show_label= False)

model_details = details

demo = gr.TabbedInterface([chat_bot, model_details], 
                          ['Chatbot', 'Model Details'],
                          theme= 'gradio/soft')


if __name__ == '__main__':
    demo.launch()