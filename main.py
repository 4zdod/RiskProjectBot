import telebot
from telebot import types
import sys
import logging
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

TOKEN = ''
bot = telebot.TeleBot(TOKEN)

user_data = {}
session_data = {}

class Asset:
    def __init__(self, name=None, loss=0, probability=0):
        self.name = name
        self.loss = loss
        self.probability = probability

@bot.message_handler(commands=['calculate'])
def start_calculate(message):
    msg = bot.send_message(message.chat.id, 'How many assets would you like to calculate?')
    bot.register_next_step_handler(msg, process_asset_count)

def process_asset_count(message):
    try:
        asset_count = int(message.text)
        user_data[message.chat.id] = {'count': asset_count, 'assets': []}
        request_asset_name(message, 0)
    except ValueError:
        bot.send_message(message.chat.id, 'You entered not a valid number, try again')

def request_asset_name(message, asset_index):
    msg = bot.send_message(message.chat.id, f'Enter the name for Asset {asset_index + 1}:')
    bot.register_next_step_handler(msg, process_asset_name, asset_index)

def process_asset_name(message, asset_index):
    asset = Asset()
    asset.name = message.text
    user_data[message.chat.id]['assets'].append(asset)
    request_asset_loss(message, asset_index)

def request_asset_loss(message, asset_index):
    msg = bot.send_message(message.chat.id, f'Enter the loss for {user_data[message.chat.id]["assets"][asset_index].name}:')
    bot.register_next_step_handler(msg, process_asset_loss, asset_index)

def process_asset_loss(message, asset_index):
    try:
        loss = int(message.text)
        user_data[message.chat.id]['assets'][asset_index].loss = loss
        request_asset_probability(message, asset_index)
    except ValueError:
        bot.send_message(message.chat.id, 'You entered not a valid number, try again')

def request_asset_probability(message, asset_index):
    msg = bot.send_message(message.chat.id, f'Enter the probability for {user_data[message.chat.id]["assets"][asset_index].name}:')
    bot.register_next_step_handler(msg, process_asset_probability, asset_index)

def process_asset_probability(message, asset_index):
    try:
        probability = float(message.text)
        user_data[message.chat.id]['assets'][asset_index].probability = probability
        if asset_index + 1 < user_data[message.chat.id]['count']:
            request_asset_name(message, asset_index + 1)
        else:
            calculate_and_send_results(message)
    except ValueError:
        bot.send_message(message.chat.id, 'You entered not a valid number, try again')

def calculate_risk_metrics(assets):
    Mx = sum(asset.loss * asset.probability for asset in assets)
    Dx = sum(asset.probability * (asset.loss - Mx) ** 2 for asset in assets)
    sigma = Dx ** 0.5
    E = 0.3  # Value specified by the condition
    R = Mx * E + ((1 - E) * sigma)
    return Mx, Dx, sigma, R

def calculate_and_send_results(message):
    assets = user_data[message.chat.id]['assets']

    # Calculate total probability of provided assets
    total_probability = sum(asset.probability for asset in assets)

    # Add "Everything will be fine" scenario
    no_risk_probability = 1 - total_probability
    assets.append(Asset(name="Everything will be fine", loss=0, probability=no_risk_probability))

    Mx, Dx, sigma, R = calculate_risk_metrics(assets)

    # Formatting the results
    results = [
        f'🔴Average Losses from Risk (Mx): {Mx:.2f}!',
        f'🟡Variance of the Result (Dx): {Dx:.2f}',
        f'🟢RMS Loss (σ): {sigma:.2f}',
        f'🔵Integral Risk Assessment (R): {R:.2f}',
        f'🟣Probability of Everything Being Fine: {no_risk_probability:.2%}'
    ]

    bot.send_message(message.chat.id, '\n'.join(results))
    del user_data[message.chat.id]

# Command handler for /ask_gpt
@bot.message_handler(commands=['ask_gpt'])
def ask_gpt_command(message):
    msg = bot.send_message(message.chat.id, 'Please enter your question for GPT:')
    bot.register_next_step_handler(msg, process_gpt_query)

def process_gpt_query(message):
    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # Load documents and initialize models
    documents = SimpleDirectoryReader("data").load_data()

    llm = LlamaCPP(
        model_url='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        model_path=None,
        temperature=0.1,
        max_new_tokens=300,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    user_question = message.text

    # Use the index to find a response
    query_engine = index.as_query_engine()
    response = query_engine.query(user_question)

    # Send the response back to the user
    bot.reply_to(message, response)


@bot.message_handler(commands=['asset_value'])
def initiate_assessment(message):
    prompt_message = bot.send_message(message.chat.id, 'Enter the number of assets for evaluation:')
    bot.register_next_step_handler(prompt_message, collect_asset_count)


def collect_asset_count(message):
    try:
        asset_quantity = int(message.text)
        session_data[message.chat.id] = {'asset_qty': asset_quantity, 'asset_details': []}
        gather_asset_details(message, 0)
    except ValueError:
        bot.send_message(message.chat.id, 'Invalid number. Please enter a valid integer.')


def gather_asset_details(message, index):
    prompt_message = bot.send_message(
        message.chat.id,
        f'Provide details for Asset {index + 1} in this format: Name,Confidentiality,Integrity,Availability,Weight (e.g., "Web Server,3,3,2,2"):'
    )
    bot.register_next_step_handler(prompt_message, store_asset_details, index)


def store_asset_details(message, index):
    try:
        parts = message.text.split(',')
        asset_entry = {
            'asset_name': parts[0],
            'confidentiality': int(parts[1]),
            'integrity': int(parts[2]),
            'availability': int(parts[3]),
            'weight': int(parts[4]),
        }
        asset_entry['asset_value'] = asset_entry['confidentiality'] + asset_entry['integrity'] + asset_entry[
            'availability']
        asset_entry['total_value'] = asset_entry['asset_value'] * asset_entry['weight']
        session_data[message.chat.id]['asset_details'].append(asset_entry)

        if index + 1 < session_data[message.chat.id]['asset_qty']:
            gather_asset_details(message, index + 1)
        else:
            conclude_assessment(message)
    except (ValueError, IndexError):
        bot.send_message(message.chat.id, 'Error in data format. Please follow the given format and try again.')


def conclude_assessment(message):
    asset_report = []
    overall_total_value = 0
    classification = {
        'Category I': [],
        'Category II': [],
        'Category III': []
    }

    for asset in session_data[message.chat.id]['asset_details']:
        asset_report.append(
            f"{asset['asset_name']} - Asset Value: {asset['asset_value']}, Total Asset Value: {asset['total_value']}"
        )
        overall_total_value += asset['total_value']
        # Categorize the asset based on its total value
        if asset['total_value'] >= 20:
            classification['Category I'].append(asset['asset_name'])
        elif 12 <= asset['total_value'] < 20:
            classification['Category II'].append(asset['asset_name'])
        else:
            classification['Category III'].append(asset['asset_name'])

    # Compile the final report
    final_report = "\n".join(asset_report) + "\n\n" + "Overall Total Asset Value: " + str(
        overall_total_value) + "\n\n" + \
                   "Categorization:\n" + \
                   f"🔴Category I Assets (Value 20+): {', '.join(classification['Category I'])}\n" + \
                   f"🟡Category II Assets (Value 12-19): {', '.join(classification['Category II'])}\n" + \
                   f"🟢Category III Assets (Value 11 or less): {', '.join(classification['Category III'])}"

    bot.send_message(message.chat.id, final_report)
    del session_data[message.chat.id]

if __name__ == "__main__":
    bot.polling(none_stop=True)
