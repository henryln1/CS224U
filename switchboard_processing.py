## SWITCHBOARD PROCESSING
import os
import csv
import re

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

caller_data = get_caller_data()
convo_data = get_conversation_data()
convos = get_conversations(convo_data)