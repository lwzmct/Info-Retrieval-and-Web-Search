## Submission.py for COMP6714-Project2
#Written by Wenzheng Li for COMP6714
###################################################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tempfile import gettempdir

from six.moves import urllib
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE



import os
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import gensim
import re

def build_dataset(words, n_words):
    """Process raw inputs into a dataset. 
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [[('UNK', 'UNK'), -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# the variable is abused in this implementation. 
# Outside the sample generation loop, it is the position of the sliding window: from data_index to data_index + span
# Inside the sample generation loop, it is the next word to be added to a size-limited buffer. 
data_index = 0
def generate_batch(batch_size, num_samples, skip_window, data, reverse_dictionary):
    global data_index

    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
    
    print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w][0] for w in buffer]))

    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        #modify
        #find each Word frequency
        sum_nb = 0
        for nb in range(span):
        	if nb != skip_window:
        		sum_nb += buffer[nb]
        join_possibility = []
        for nb in range(span):
        	if nb == skip_window:
        		join_possibility.append(0)
        	else:
        		if sum_nb == 0:
        			join_possibility.append(1 / (span - 1))
        		else:
        			join_possibility.append(buffer[nb] / sum_nb)
        word_to_use_new = collections.deque(maxlen=num_samples)
        random_nb = random.uniform(0,1)
        for p in range(span):
        	s = random_nb + join_possibility[p]
        	if s > 1:
        		word_to_use_new.append(p)
        	if len(word_to_use_new) >= num_samples:
        		break
        #check length
        random.shuffle(context_words)
        while len(word_to_use_new) < num_samples:
        	word_to_use_new.append(context_words.pop())
        	if len(context_words) == 0:
        		break

        #random.shuffle(context_words)
        #words_to_use = collections.deque(context_words) # now we obtain a random list of context words
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = word_to_use_new.pop()
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
        
        # slide the window to the next position    
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else: 
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
        
        print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w][0] for w in buffer]))
        
    # end-of-for
    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels




def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    #print(count[:10], data[:10])
    batch_size = 117
    skip_window = 2
    num_samples = 3
    num_sampled = 64
    vocabulary_size = 4000
    logs_path = './log2/'

    #read file
    data_list = list()
    d_file = open(data_file, 'r')
    this_txt = d_file.read()
    pre_data_list = re.split(r'[\s]', this_txt)
    for i in range(0, len(pre_data_list) - 1, 2):
        data_list.append((pre_data_list[i], pre_data_list[i + 1]))
    #print(data_list)
    
    data, count, dictionary, reverse_dictionary = build_dataset(data_list, vocabulary_size)

    sample_size = 20
    sample_window = 100
    sample_examples = np.random.choice(sample_window, sample_size, replace=False)
    
    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            with tf.name_scope('Embeddings'):
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim],
                                                              stddev=1.0 / math.sqrt(embedding_dim)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, labels=train_labels,
                                                     inputs=embed, num_sampled=num_sampled,
                                                     num_classes=vocabulary_size))
            with tf.name_scope('Gradient_Descent'):
                optimizer = tf.train.AdamOptimizer(learning_rate = 0.003).minimize(loss)
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
            
            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)
        
            # Add variable initializer.
            init = tf.global_variables_initializer()
            
            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()
            
    #training
    with tf.Session(graph=graph) as session:
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window,
                                                        data, reverse_dictionary)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
            
            summary_writer.add_summary(summary, step)
            average_loss += loss_val
            
            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
                    
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]][0]#modified
                    top_k = 10
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    #print(nearest)
                    #print(len(reverse_dictionary))
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]][0]#modified
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                print()
                
        final_embeddings = normalized_embeddings.eval()

    #print(final_embeddings[:5])
    #print('1 final_embeddings',len(final_embeddings))
    #print(len(final_embeddings[1]))
    #print('2 reverse_dictionary',len(reverse_dictionary))
    #print('3',len(data_file))
    f = open(embeddings_file_name, 'w')
    f.write(str(len(reverse_dictionary)))
    f.write(' ')
    f.write(str(embedding_dim))
    f.write('\n')
    for i in range(len(reverse_dictionary)):
        this_line = str(reverse_dictionary[i][0])+'_'+str(reverse_dictionary[i][1])#modify
        for j in final_embeddings[i]:
            this_line = this_line + ' ' + str(j)
        this_line += '\n'
        f.write(this_line)
    f.close()



def process_data(input_data):
    # Remove this pass line, you need to implement your code to process data here...
    nlp = spacy.load('en')
    org_txt = list()
    with zipfile.ZipFile(input_data) as f:
        t = len(f.namelist())
        for i in range(t):
            test_doc = tf.compat.as_str(f.read(f.namelist()[i]))
            test_doc = nlp(test_doc)
            for token in test_doc:
                if token.is_alpha:
                    if token.ent_iob == 2:
                        if token.pos == 88:
                            org_txt.append(('the', 'DET'))
                        elif token.pos == 98 or token.pos == 90:
                            org_txt.append((str(token.lemma_).lower(), str(token.pos_)))
                        elif token.pos == 83:
                            org_txt.append(('ADP', 'ADP'))
                        elif token.pos == 93:
                            org_txt.append(('PRON', 'PRON'))
                        elif token.pos == 87:
                            org_txt.append(('and', 'CCONJ'))
                        elif token.pos == 92:
                            org_txt.append(('to', 'PART'))
                        elif str(token).lower() in ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
                        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                        'w', 'x', 'y', 'z', 'km', 'wi', 'uk', 'fa', 'ca', 'us', 'ds']:
                            org_txt.append(('Single_letter', 'NOUN'))
                        elif str(token).lower() in ['which', 'that', 'whose', 'what', 'how',
                        'this', 'that\'s', 'whom', 'how']:
                            org_txt.append(('that','ADP'))
                        else:
                            org_txt.append((str(token).lower(), str(token.pos_)))
                    else:
                        org_txt.append(('Ent', 'NOUN'))
    #consider m
    f = open('output_data_file.txt', 'w')
    for i in org_txt:
        f.write(i[0])
        f.write(' ')
        f.write(i[1])
        f.write(' ')
    f.close()
    return 'output_data_file.txt'
#a = process_data('./BBC_Data.zip')
#print(a[:200])

def Compute_topk(model_file, input_adjective, top_k):
    # Remove this pass line, you need to implement your code to compute top_k words similar to input_adjective
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    if input_adjective+'_ADJ' not in model:
    	return []
    result_list = []
    result_list = model.most_similar(positive = [input_adjective + '_ADJ'], topn = 10 * top_k)
    #print(len(result_list))
    return_list = []
    for i in range(10 * top_k):
        if result_list[i][0][-3:] == 'ADJ':
            return_list.append(result_list[i][0][:-4])
        #return_list.append(result_list[i][0])
        if len(return_list) == top_k:
            break
    return return_list

def Evaluation_result():
    model_file = 'adjective_embeddings.txt'
    k = 100
    file_index = 0
    path = './dev_set'
    files = os.listdir(path)
    hits = 0
    for file_name in files:
        print(file_index)
        file_index += 1
        ground_truth = 'Ground_truth_' + file_name + ' = '
        f = open(path + '/' + file_name)
        true_word_list = []
        for line in f:
            true_word_list.append(line[:-1])
        ground_truth += str(true_word_list)
        print(ground_truth)
        output_result = 'Output_' + file_name + ' = '
        output = []
        output = Compute_topk(model_file, file_name, k)
        if output == []:
        	output_result += str(output)
        else:
        	output_result += str(output[:k])
        print(output_result)
        hit = 0
        if output == []:
        	hit = 0
        else:
        	for i in output[:k]:
        		if i in true_word_list:
        			hit += 1
        print('hit: ', hit)
        hits += hit
    print('k = ',k,' average_hit: ', hits / file_index)




# input_dir = './BBC_Data.zip'
# data_file = process_data(input_dir)
# embedding_file_name = 'adjective_embeddings.txt'
# num_steps = 100001
# embedding_dim = 200
# adjective_embeddings(data_file, embedding_file_name, num_steps, embedding_dim)
# model_file = 'adjective_embeddings.txt'
# #model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
# model_file = 'adjective_embeddings.txt'
# input_adjective = 'bad'
# top_k = 5
# output = []
# output = Compute_topk(model_file, input_adjective, top_k)
# print(output)

def main():
    input_dir = './BBC_Data.zip'
    data_file = process_data(input_dir)
    embedding_file_name = 'adjective_embeddings.txt'
    num_steps = 100001
    embedding_dim = 200
    adjective_embeddings(data_file, embedding_file_name, num_steps, embedding_dim)
    #model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    #model_file = 'adjective_embeddings.txt'
    #input_adjective = 'bad'
    #top_k = 5
    #output = []
    #output = Compute_topk(model_file, input_adjective, top_k)
    #print(output)
    Evaluation_result()

if __name__ == '__main__':
    main()
