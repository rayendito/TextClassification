import pandas as pd

def build_alay_dict():
    df = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')
    return dict(zip(df['slang'], df['formal']))

def translate_alay(input_string, alay_dict):
    string_splitted = input_string.split(" ")
    for i in range(len(string_splitted)):
        if(string_splitted[i] in alay_dict):
            string_splitted[i] = alay_dict[string_splitted[i]]
    return ' '.join(string_splitted)