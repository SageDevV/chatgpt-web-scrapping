# Gerais
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# Scraping
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from urllib.request import Request, urlopen

# Interface
import gradio as gr

# Assistente
import tiktoken
import openai
import time
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity


urls = ['https://small-brain-54b.notion.site/Page-for-web-scrapping-fe0f087e53d647f8b6cd3dea206580a7?pvs=4']

domain = "small-brain-54b.notion.site"

url = "https://small-brain-54b.notion.site/Page-for-web-scrapping-fe0f087e53d647f8b6cd3dea206580a7?pvs=4"

def get_text_from_url(url):
    
    # Analisa a URL e pega o domínio
    local_domain = urlparse(url).netloc
    print(local_domain)

    # Fila para armazenar as urls para fazer o scraping
    fila = deque(urls)
    print(fila)

    # Criar um diretório para armazenar os arquivos de texto
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # Enquanto a fila não estiver vazia, continue fazendo o scraping
    while fila:
        # Pega a próxima URL da fila
        url = fila.pop()
        print("Próxima url",url) # Checa próxima url

        # Salva o texto da url em um arquivo <url>.txt
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w") as f:
            driver = Chrome()
            driver.get(url)
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            text = page_soup.get_text()
            f.write(text)

        driver.close()

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def create_context(question, df, max_len=1800, size="ada"):

    """
    Cria um contexto para uma pergunta encontrando o contexto mais similar no conjunto de embeddings gerado utilizando o Custom Knowledge.
    """

    # Obter a embeddings para a pergunta que foi feita
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Obter as distâncias a partir dos embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Classifique por distância e adicione o texto ao contexto
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Adicionar o comprimento do texto ao comprimento atual
        cur_len += row['n_tokens'] + 4
        
        # Se o contexto for muito longo, quebre
        if cur_len > max_len:
            break
        
        # Caso contrário, adicione-o ao texto que está sendo retornado
        returns.append(row["text"])

    # Retornar o contexto
    return "\n\n###\n\n".join(returns)

def answer_question(
                    df=df,
                    model="text-davinci-003",
                    question="O que é a bix tecnologia?",
                    max_len=1800,
                    size="ada",
                    debug=False,
                    max_tokens=150,
                    stop_sequence=None):
    """
    Responder a uma pergunta com base no contexto mais semelhante dos textos do dataframe
    """
    context = create_context(
        question,
        df=df,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Criar uma conclusão usando a pergunta e o contexto
        response = openai.Completion.create(
            prompt=f"Responda as perguntas com base no contexto abaixo, e se a pergunta não puder ser respondida diga \"Eu não sei responder isso\"\n\Contexto: {context}\n\n---\n\nPergunta: {question}\nResposta:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)

def chatgpt_clone(input, history):
    history= history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output=answer_question(question = inp)
    history.append((input, output))
    return history, history

get_text_from_url(url)

texts=[]
for file in os.listdir("text/" + domain + "/"):

    with open("text/" + domain + "/" + file, "r") as f:
        text = f.read()

        texts.append((file[20:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))


df = pd.DataFrame(texts, columns = ['page_name', 'text'])

df['text'] = df.page_name + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

print("Checando número de páginas extraidas e urls informadas \n")
print("Número de páginas",df.shape[0])
print("\nNúmero de urls informadas",len(urls))

tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']


df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))


df.hist(column='n_tokens')

df.head()

max_tokens = 500


def split_into_many(text, max_tokens = max_tokens):

    sentences = text.split('. ')

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):

        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

shortened = []

for row in df.iterrows():

    if row[1]['text'] is None:
        continue

    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    else:
        shortened.append( row[1]['text'] )

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.hist(column='n_tokens')

num_tot_tokens = df['n_tokens'].sum()
print("Número total de tokens",num_tot_tokens)

df.head()

i = 0
for text in df['text']:
    i+=1
    
print("Número de trechos de texto com no máximo",max_tokens,"tokens :",i)

print("Custo total de treinamento do embedding: $",num_tot_tokens /1000 * 0.0001)

def read_openai_api_key():
    with open('openai_secret.txt', 'r') as file:
        api_key = file.read().strip()
    return api_key

my_api_key = read_openai_api_key()

openai.api_key = read_openai_api_key()

i = 0
embeddings = []
for text in df['text']:
    time.sleep(2)
    print(i)
    try:
        embedding = openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
        print("Fazendo embedding do texto")
        embeddings.append(embedding)
    except openai.error.RateLimitError:
        print("Rate limit error, esperando 20 segundo antes de tentar novamente")
        time.sleep(20)  
        embedding = openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
        print("embedding texto depois de esperar 20 segundos")
        embeddings.append(embedding)
    i+=1

df['embeddings'] = embeddings
df.to_csv('processed/embeddings.csv')
df.head()

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()

answer_question(question="o que está escrito nas 3 primeiras linhas?", debug=False)

answer_question(question="o que está escrito nas 3 ultimas linhas?", debug=False)

with gr.Blocks(theme=gr.themes.Soft(),css=".gradio-container {background-color: lightsteelblue}") as block:
    with gr.Row():
        img1 = gr.Image("images/BIX_branding-9.png",show_label=False, width=100, height=100)
        img2 = gr.Image("images/Logo Azul_Vertical.png",show_label=False, width=100, height=100)
    gr.Markdown("""<h1><center> Assistente da Bix Tecnologia</center></h1>""")
    chatbot=gr.Chatbot(label="Conversa")
    message=gr.Textbox(label="Faça sua pergunta",placeholder="O que você gostaria de saber sobre a Bix Tecnologia?")
    state = gr.State()
    submit = gr.Button("Perguntar")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])