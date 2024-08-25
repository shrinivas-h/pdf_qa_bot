import pickle
import re
import json


def remove_non_ascii(input_string):
    return input_string.encode('ascii', 'ignore').decode('ascii')


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_json_file(file_path, new_data):
    with open(file_path, 'w') as file:
        file.write(json.dumps(new_data))


def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_pickle_file(file_path, data):
    with open(file_path, 'wb') as file:
        file.write(pickle.dumps(data))


def remove_images_and_links_from_text(text):
    data = read_json_file("common/resources/text_processing_config.json")

    for regex in data['regular_expressions']["link_image_regular_expressions"]:
        text = re.sub(regex, '', text)

    text = text.strip()
    return text