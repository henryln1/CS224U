## SWITCHBOARD PROCESSING
import os
import csv
import re
import random

switchboard_directory = 'switchboard-corpus/'

switchboard_folders = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

caller_file = 'caller_tab.csv'

conversation_file = 'conv_tab.csv'

# returns dict {convo # -> (person A #, person B #)}
def get_conversation_data():
    conversations = {}
    with open(switchboard_directory + conversation_file) as csvfile:
        filereader = csv.reader(csvfile, delimiter = ',')
        for row in filereader:
            conversations[row[0].strip()] = (row[2].strip(), row[3].strip())
    return conversations

# returns dict {person # -> gender}
def get_caller_data():
    callers = {}
    with open(switchboard_directory + caller_file) as csvfile:
        filereader = csv.reader(csvfile, delimiter = ',', quotechar='"')
        for row in filereader:
            callers[row[0].strip()] = row[3].strip('" ')
    return callers

# helper function to get rid of unneeded transcription notation
# this was based on a small sample of the conversations - might need to be expanded
def process_dialog(dialog):
    dialog = re.sub('{[A-Z]', '', dialog)
    dialog = re.sub('<[Ll]aughter>', 'haha', dialog) # need to standardize movie dialog laughter too I think
    dialog = re.sub('<+[A-Za-z _]+>+', '', dialog)
    dialog = re.sub('[}\[\]\+/\(\)#]', '', dialog)
    dialog = re.sub('  +', ' ', dialog)
    dialog = dialog.strip()
    return dialog
                

# returns dict {convo # -> list of lines}
# list of lines contains (person #, line) for each line
def get_conversations(convo_data):
    conversations = {}
    for folder in switchboard_folders:
        directory = switchboard_directory + 'sw' + folder + 'utt/'
        for filename in os.listdir(directory):
            if filename == 'words':
                continue
            conversation_num = filename[-8:-4] # theres also a line in each file that's "FILENAME: (convo num)_(person a num)_(person b num)" not sure if I should just use that instead
            personA, personB = convo_data[conversation_num]
            lines = []
            with open(directory + filename) as f:
                for line in f:
                    match = re.search('([AB])\.\d+ utt\d+:(.+)', line)
                    if match == None:
                        continue
                    identifier = personA if match.group(1) == 'A' else personB
                    dialog = match.group(2).strip()
                    dialog = process_dialog(dialog)
                    lines.append((identifier, dialog))
            conversations[conversation_num] = lines
    return conversations

# returns dict {topic # -> list of lines}
# list of lines contains (person #, line) for each l ine
def get_topics_conversations(convo_data):
    conversations = {}
    for folder in switchboard_folders:
        directory = switchboard_directory + 'sw' + folder + 'utt/'
        for filename in os.listdir(directory):
            if filename == 'words':
                continue
            conversation_num = filename[-8:-4] # theres also a line in each file that's "FILENAME: (convo num)_(person a num)_(person b num)" not sure if I should just use that instead
            personA, personB = convo_data[conversation_num]
            lines = []
            topic_num = -1
            with open(directory + filename) as f:
                for line in f:
                    topic_match = re.search('TOPIC#:\s*(\d+)', line)
                    if topic_match != None:
                        topic_num = topic_match.group(1)
                    match = re.search('([AB])\.\d+ utt\d+:(.+)', line)
                    if match == None:
                        continue
                    identifier = personA if match.group(1) == 'A' else personB
                    dialog = match.group(2).strip()
                    dialog = process_dialog(dialog)
                    lines.append((identifier, dialog))
            if topic_num not in conversations:
                conversations[topic_num] = lines
            else:
                conversations[topic_num] += lines
    return conversations

# returns list of all lines and all associated genders
def get_labeled_lines(convos, people):
    lines = []
    genders = []
    for number, convo in convos.items():
        for person, line in convo:
            lines.append(line.lower())
            genders.append(people[person][0].lower())
    return lines, genders

# same as above, but returns equal num of male and female lines
# there are 92887 male lines and 130729 female lines, so we take 90000 of each
# this does not take into account conversation topics
def get_labeled_lines_equal(convos, people):
    size = 90000
    male_lines = []
    female_lines = []
    for number, convo in convos.items():
        for person, line in convo:
            if people[person][0].lower() == 'f':
                female_lines.append(line.lower())
            else:
                male_lines.append(line.lower())
    female_equal = random.sample(female_lines, size)
    male_equal = random.sample(male_lines, size)
    lines = female_equal + male_equal
    genders = ['f' if x < size else 'm' for x in range(size * 2)]
    return lines, genders

def get_labeled_lines_equal_topic(convos, people):
    lines = []
    genders = []
    for topic, convo in convos.items():
        male_lines = []
        female_lines = []
        for person, line in convo:
            if len(line.split()) < 10:
                continue
            if people[person][0].lower() == 'f':
                female_lines.append(line.lower())
            else:
                male_lines.append(line.lower())
        num_lines = min([len(male_lines), len(female_lines)])
        female_equal = random.sample(female_lines, num_lines)
        male_equal = random.sample(male_lines, num_lines)
        lines += female_equal + male_equal
        genders += ['f' if x < num_lines else 'm' for x in range(num_lines * 2)]
    return lines, genders


def get_switchboard_data():
    caller_data = get_caller_data()
    convo_data = get_conversation_data()
    # convos = get_conversations(convo_data)
    # labeled = get_labeled_lines_equal(convos, caller_data)
    convos = get_topics_conversations(convo_data)
    labeled = get_labeled_lines_equal_topic(convos, caller_data)
    return labeled