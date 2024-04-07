import configparser
import re
import ast
import openai
import time
from IPython.display import display, Markdown

def load_api_key_from_file(config_file='../config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    api_key = config['DEFAULT']['API_KEY']
    if not api_key:
        raise ValueError("API key not found in the configuration file")
    return api_key

def chatGPT_replay(client,thread_id,assistant,question_content):
    
    message = client.beta.threads.messages.create(
      thread_id=thread_id,
      role="user",
      content=question_content,
      file_ids=assistant.file_ids
    )
    
    run = client.beta.threads.runs.create(
      thread_id=thread_id,
      assistant_id=assistant.id
    )
    return run

def chatGPT_check_replay(client,thread,dis=True):
    print('start to check the process')
    runs = client.beta.threads.runs.list(
          thread.id)
    
    while runs.data[0].status != 'completed':
        time.sleep(3)
        runs = client.beta.threads.runs.list(
          thread.id
        )
    if dis:
        thread_messages = client.beta.threads.messages.list(thread.id)
        display(Markdown(thread_messages.data[0].content[0].text.value))
    return 

def extract_code_script_from_markdown(file):

#     pattern = r'```python\n(.*?)\n```'
    pattern = r"```python\n(.*?)\n```"

    # Find all matches; using re.DOTALL to make '.' match newlines
    matches = re.findall(pattern, file, re.DOTALL)

    if matches:
        return matches
    else:
        return "No Python dictionary found in the markdown file"

def find_dictionaries_in_string(s):
    # Regular expression to match dictionary patterns
    # This pattern looks for strings that start with '{', end with '}', and contain key-value pairs
    pattern = r"\{.*?\}"
    
    # Find potential dictionary strings
    potential_dicts = re.findall(pattern, s, re.DOTALL)

    # Try to parse each potential dictionary
    extracted_dicts = []
    for dict_str in potential_dicts:
        try:
            # Safely evaluate the string to see if it's a valid dictionary
            result_dict = eval(dict_str)
            if isinstance(result_dict, dict):  # Check if the result is indeed a dictionary
                extracted_dicts.append(result_dict)
        except Exception:
            # If an error occurs, skip this string
            continue

    return extracted_dicts
