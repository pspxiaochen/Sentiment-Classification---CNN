import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore')
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle

positive = pickle.load(open('process_review/phone_positive_2.txt', 'rb'))
negative = pickle.load(open('process_review/phone_negative_2.txt', 'rb'))

with open('cellphone_feature_words.txt','r') as infile:
    phone_feature_words = [l.strip('\n') for l in infile.readlines()]

dict_cellphone = {}
for i in range(len(phone_feature_words)):
    dict_cellphone[phone_feature_words[i]] = np.random.uniform(low=-1, high=1, size=50)

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
batch_size = 30

def get_all_triple_review_and_aspect(test_positive,test_negative):
    all_triple_review = []
    all_triple_aspect = []
    j = 0
    for i in range(0,len(test_positive.review),2):
        if j >= len(test_negative)-1:
            break
        triple_review = []
        triple_aspect = []
        triple_review.append([test_positive.review[i]])
        triple_aspect.append([test_positive.aspects[i]])
        triple_review.append([test_positive.review[i+1]])
        triple_aspect.append([test_positive.aspects[i+1]])
        triple_review.append([test_negative.review[j]])
        triple_aspect.append([test_negative.aspects[j]])
        all_triple_review.append(triple_review)
        all_triple_aspect.append(triple_aspect)
        j += 1
        print('已经生成了', j, '个三元组')
    return all_triple_review,all_triple_aspect
all_triple_review,all_triple_aspect = get_all_triple_review_and_aspect(positive[:200],negative[:100])

# 使三元组的数量能够整除batch_size
def get_batch_size_triple(triple):
    ex = len(triple) % batch_size
    if ex != 0:
        triple = triple[:len(triple)-ex]
    return triple

all_triple_review = get_batch_size_triple(all_triple_review)
all_triple_aspect = get_batch_size_triple(all_triple_aspect)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
embedding_size = 300

def _get_word_vecs(review):
    vecs = []
    review = np.array(review)
    for i in range(len(review)):
        for j in range(len(review[i])):
            for k in range(len(review[i][j])):
                for word in review[i][j][k]:
                    try:
                        vecs.append(model[word])
                    except KeyError:
                        vecs.append(np.random.uniform(low=-1, high=1, size=(embedding_size)))
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype='float')

def _get_aspect_vecs(aspect):
    vecs = []
    for i in range(len(aspect)):
        for j in range(len(aspect[i])):
            for k in range(len(aspect[i][j])):
                vecs.append(dict_cellphone[aspect[i][j][k]])
    vecs = np.concatenate(vecs)
    return np.array(vecs,dtype='float')

#save_model_path = './save_model'

import tensorflow as tf
# Placeholders for input, output and dropout
def neural_net_input():
    input_x = tf.placeholder(tf.float32,[None, sequence_length, embedding_size, 1],name="input_x")
    return input_x

def neural_net_aspect():
    aspects_tensor = tf.placeholder(tf.float32,[None,50],name="aspect_tensor")
    return aspects_tensor

# Convolution layer and max pooling layer
def conv2d_maxpool(input_x,filter_sizes,num_filters):
    pooled_outputs= []
    for i,filter_size in enumerate(filter_sizes):
    # Convolution layer
        filter_shape = [filter_size,embedding_size,1,num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
        b = tf.Variable(tf.truncated_normal(shape=[num_filters],stddev=0.1),name="b")
        conv = tf.nn.conv2d(input_x,W,strides=[1,1,1,1],padding="VALID",name="conv")
        h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="tanh")
    #pooling
        pooled = tf.nn.max_pool(h,ksize=[1,sequence_length - filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name="pool")
        pooled_outputs.append(pooled)
    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes) #600
    h_pool = tf.concat(values=pooled_outputs,axis=3)
    h_pool_flat = tf.reshape(h_pool,[-1,num_filters_total])
    return h_pool_flat,num_filters_total

def maxpool_to_hidden(h_pool_flat,num_filters_total):
    W = tf.Variable(tf.truncated_normal([num_filters_total,300],stddev=0.1),name="W")
    b = tf.Variable(tf.truncated_normal(shape=[300],stddev=0.1),name="b")
    out = tf.nn.tanh(tf.nn.xw_plus_b(h_pool_flat,W,b),name="out")
    return out

def h_add_aspect(input,aspect_word):
    out = tf.concat(values=[input,aspect_word],axis=1)
    return out

def embedding_layer(input):
    W = tf.Variable(tf.truncated_normal([350,300],stddev=0.1),name="W")
    b = tf.Variable(tf.truncated_normal(shape=[300],stddev=0.1),name="b")
    out = tf.nn.tanh(tf.nn.xw_plus_b(input,W,b),name="out")
    return out

def conv_net(input_x,aspect_word):
    h_pool_flat,num_filters_total = conv2d_maxpool(input_x,[1,2,3],200)
    out = maxpool_to_hidden(h_pool_flat,num_filters_total)
    out = h_add_aspect(out,aspect_word)
    out =embedding_layer(out)
    return out

def dst(s1,s2):
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(s1-s2),0))
    return euclidean

def get_loss(lamda,logits,batch_size):
    with tf.name_scope("loss"):
        loss = 0
        for i in range(0,batch_size*3,3):
            loss_ = tf.maximum(0.0,lamda - dst(logits[i],logits[i+2]) + dst(logits[i],logits[i+1]),name='loss')
            loss += loss_
        tf.summary.scalar('loss',loss)
        return loss

epochs =200
tf.reset_default_graph()
input_x = neural_net_input()
aspects_tensor = neural_net_aspect()
logits = conv_net(input_x,aspects_tensor)
logits = tf.identity(logits,name="logits")
lamda = 5.0
loss = get_loss(lamda,logits,batch_size)
tf.summary.scalar('loss',loss)
learning_rate = 0.002
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def train_neural_network(session,optimizer,feature_batch,aspects_tensor_):
    session.run(optimizer,feed_dict={input_x:feature_batch,aspects_tensor:aspects_tensor_})

def print_stats(session,feature_batch,loss,aspects_tensor_,bitch_size):
    feed_dict = {input_x:feature_batch,aspects_tensor:aspects_tensor_}
    loss_batch = session.run(loss,feed_dict=feed_dict)
    return loss_batch

print('Training...')

with tf.Session() as sess:
    total_loss = 0
    j = 0
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs',sess.graph)
    for epoch in range(epochs):
        for r in range(0,len(all_triple_review),batch_size):
            review = _get_word_vecs(all_triple_review[r:r+batch_size]).reshape(batch_size,3,sequence_length,embedding_size,1)
            aspect = _get_aspect_vecs(all_triple_aspect[r:r+batch_size]).reshape(batch_size,3,50)
            review_vecs_batch = review.reshape(batch_size*review.shape[1],review.shape[2],review.shape[3],review.shape[4])
            aspect_vecs_batch = aspect.reshape(batch_size*aspect.shape[1],aspect.shape[2])
            train_neural_network(sess,optimizer,review_vecs_batch,aspect_vecs_batch)
            loss_= print_stats(sess,review_vecs_batch,loss,aspect_vecs_batch,batch_size)
            print('epoch=', epoch, '第', r + 1, '个三元组的loss:' + str(loss_))
            rs = sess.run(merged,feed_dict={input_x:review_vecs_batch,aspects_tensor:aspect_vecs_batch})
            writer.add_summary(rs)
            #total_loss += loss_
    #Save Model
    # saver = tf.train.Saver()
    # save_path = saver.save(sess,save_model_path)













