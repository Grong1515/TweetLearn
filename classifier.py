import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import pymorphy2


BANK_TRAIN_BASE = 'data/bank_train_2016.xml'
BANK_TEST_BASE = 'data/banks_test_etalon.xml'
TKK_TRAIN_BASE = 'data/tkk_train_2016.xml'
TKK_TEST_BASE = 'data/tkk_test_etalon.xml'

BANK_LIST = ['rshb', 'alfabank', 'vtb', 'bankmoskvy', 'raiffeisen', 'sberbank', 'uralsib', 'gazprom']


def load_data(file, names):
    tree = ET.parse(file)
    database = tree.getroot().find('database')

    train_set = []
    answer_set = []

    for table in database.iter('table'):
        train_set.append(table.find('./*[@name="text"]').text)
        answer = set()
        for item in names:
            value = table.find('./*[@name="{}"]'.format(item)).text
            if value != "NULL":
                answer.add(value)
        answer_set.append(list(answer))
    return train_set, answer_set


def f_tokenizer(s):
    morph = pymorphy2.MorphAnalyzer()
    if type(s) == unicode:
        t = s.split(' ')
    else:
        t = s
    f = []
    for j in t:
        m = morph.parse(j.replace('.',''))
        if len(m) != 0:
            wrd = m[0]
            if wrd.tag.POS not in ('NUMR','PREP','CONJ','PRCL','INTJ'):
                f.append(wrd.normal_form)
    return f

if __name__ == '__main__':
    pass