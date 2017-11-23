import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore')
import pandas as pd
import nltk
import numpy as np
import pickle
positive = pd.read_csv('data/cell_positive.csv')
negative = pd.read_csv('data/cell_negative.csv')

with open('feature_words/cellphone_feature_words.txt','r') as infile:
    phone_feature_words = [l.strip('\n') for l in infile.readlines()]

dict_cellphone = {}
for i in range(len(phone_feature_words)):
    dict_cellphone[phone_feature_words[i]] = np.random.uniform(low=-1, high=1, size=50)

#将句子全部转化为小写
positive.review = positive.review.map(lambda review:review.lower())
negative.review = negative.review.map(lambda review:review.lower())

#对每个句子设置它的aspects词
def set_aspects(test):
    test['aspects'] = phone_feature_words[-1]
    for i in range(len(test.review)):
        test.review[i] = nltk.word_tokenize(test.review[i])
        for word in test.review[i] :
            if (word in phone_feature_words):
                test['aspects'][i] = word
                break
        print('已经设置了',i+1,'个额外词，还剩下',len(test.review)-i)
    return test

positive = set_aspects(positive)
negative = set_aspects(negative)

#得到句子最长单词数
def return_sequence_length(df):
    sequence_length = 0
    for review in df.review:
        k = len(review)
        if (k > sequence_length):
            sequence_length = k
    print('已经得到句子最大长度')
    return sequence_length
sequence_length = max(return_sequence_length(positive),return_sequence_length(negative))

#将句子都设置成同样的长度
def set_review_size(df,sequence_length):
    for i in range(len(df.review)):
        if(len(df.review[i])<sequence_length):
            df.review[i] = df.review[i] + [0]*(sequence_length-len(df.review[i]))
        print('已经处理了',i+1,'条，还剩下',len(df.review)-i,'条')
    return df

test_positive = set_review_size(positive,sequence_length)
test_negative = set_review_size(negative,sequence_length)

positive = pickle.load(open('process_review/phone_positive_2.txt', 'rb'))
negative = pickle.load(open('process_review/phone_negative_2.txt', 'rb'))