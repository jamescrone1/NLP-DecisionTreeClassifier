import spacy
import os.path
import csv

from .features import search_word, get_word_frequency_number

BASE = os.path.dirname(os.path.abspath(__file__))
nlp = spacy.load("en_core_web_lg")
file_path = 'C:\\Users\\james\\Documents\\Queens\\Final Year Project\\Dataset\\generatedFiles\\Actual\\'


def get_sentences_from_text_file():
    text_sentences = []
    with open(file_path + "temporary.txt", 'r', encoding="utf-8") as f:
        text = f.read().split('.')
        for line in text:
            line = bytes(line, 'utf-8').decode('utf-8', 'ignore')
            text_sentences.append(line)
    return text_sentences


def write_temporary_text_file(text):
    with open(file_path + "temporary.txt", 'w+', encoding="utf-8") as f:
        f.write(text)


def get_sentences_from_text(text):
    text_sentences = []
    new_text = text.split('.')
    for line in new_text:
        line = bytes(line, 'utf-8').decode('utf-8', 'ignore')
        text_sentences.append(line)
    return text_sentences


def read_data():
    text_sentences = get_sentences_from_text_file()
    sentence_processor(text_sentences)


def sentence_processor(text_sentences):
    with open(file_path + 'cv_sentence_list_3.csv', 'w+', newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence No.", "Text", "Is Entity", "Start Index", "End Index", "Entity Start Index",
                         "Entity End Index", "In Dictionary", "Word Frequency", "Is University", "Label"])
        for counter_sentence, sentence in enumerate(text_sentences):
            print(counter_sentence)
            doc = nlp(sentence)
            third_attempt(doc, counter_sentence, writer)


def sentence_processor_single_cv(text_sentences):
    with open(file_path + 'temporary.csv', 'w+', newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence No.", "Text", "Is Entity", "Start Index", "End Index", "Entity Start Index",
                         "Entity End Index", "In Dictionary", "Word Frequency", "Is University", "Label"])
        for counter_sentence, sentence in enumerate(text_sentences):
            print(counter_sentence)
            doc = nlp(sentence)
            third_attempt(doc, counter_sentence, writer)


def search_word_in_dictionary(token):
    if search_word(token.text):
        return "Yes"
    else:
        return "No"


def get_entity_length(token, doc, index, ent_start, ent_end):
    counter = 1
    entity_end_length = ent_end
    if index == len(doc)-1:
        return ent_start, entity_end_length
    while doc[index + counter].ent_type_ == token.ent_type_:
        if index <= ((len(doc)-1) - counter):
            entity_end_length = entity_end_length + len(doc[index + counter])
            if (index + counter) >= (len(doc)-1):
                return ent_start, entity_end_length
            counter = counter + 1
    return ent_start, entity_end_length


def get_entity_length_inverse(token, doc, index):
    counter = 1
    start_index = token.idx
    entity_end_length = len(token)
    if index == 0:
        return get_entity_length(token, doc, index, start_index, (entity_end_length + token.idx))
    while doc[index - counter].ent_type_ == token.ent_type_:
        start_index = doc[index - counter].idx
        if index - counter >= 0:
            entity_end_length = entity_end_length + len(doc[index - counter])
            if (index - counter) < 0:
                return get_entity_length(token, doc, index, start_index, (entity_end_length + start_index))
            counter = counter + 1
    return get_entity_length(token, doc, index, start_index, (entity_end_length + start_index))


def third_attempt(doc, counter_sentence, writer):
    words = []
    for index, token in enumerate(doc):
        if token.pos_ is not "SPACE":
            words.append(token.text)
            if token.ent_type_ is "":
                words.append("No")
            else:
                words.append(token.ent_type_)
            words.append(token.idx)
            words.append(token.idx + len(token))
            get_entity_indexes(token, doc, index, words)
            words.append(search_word_in_dictionary(token))
            words.append(get_word_frequency_number(token))
            ratio = is_university(token)
            words.append(ratio)
            words.append("")
    write_csv_row(words, counter_sentence, writer)


def get_entity_indexes(token, doc, index, words):
    if token.ent_type_ is not "":
        (entity_start, entity_end) = get_entity_length_inverse(token, doc, index)
        words.append(entity_start)
        words.append(entity_end)
    else:
        words.append("0")
        words.append("0")


def get_full_entity_word(token, doc, index):
    counter = 1
    entity_word = token.text
    if index == len(doc)-1:
        return entity_word, counter
    while doc[index + counter].ent_type_ == token.ent_type_:
        if index <= ((len(doc)-1) - counter):
            entity_word = entity_word + " " + doc[index + counter].text
            if (index + counter) >= (len(doc)-1):
                return entity_word, counter
            counter = counter + 1
    return entity_word, counter


def is_university(token):
    return read_csv_for_universities(token.text)


def write_csv_row(words, counter_sentence, writer):
    for index in range(0, len(words) - 10, 10):
        writer.writerow([counter_sentence + 1, words[index], words[index + 1], words[index + 2],
                         words[index + 3], words[index + 4],
                         words[index + 5], words[index + 6], words[index + 7], words[index + 8], words[index + 9]])

