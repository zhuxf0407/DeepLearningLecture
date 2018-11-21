'''
Author: Xiaofei Zhu, zxf@cqut.edu.cn

Date: 2018.11.15

Place: B504, 1st Lab Building, CQUT
'''
texts = ['A brief reference about the graphical user interface',
         'Python own help system',
         'Details about object use object for extra detail',
         'Introduction and overview of IPython features.',
         'Input arrays should have the same number of samples as target arrays. Found 4 input samples and 5 target samples']
labels = [0,1,2,0,2] 
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH= 4
EMBEDDING_DIM = 100
import numpy as np
np.random.seed(1234)



def load_data(texts, labels): 
    from keras.utils.np_utils import to_categorical
    from keras.preprocessing.text import Tokenizer  
    from keras.preprocessing.sequence import pad_sequences
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts) 
    word_index = tokenizer.word_index
    
    data = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='post')
    
    labels = to_categorical(labels) 
    return data, labels,word_index
data, labels,word_index = load_data(texts, labels)


def build_model():
    num_words = min(len(word_index)+1 , MAX_NB_WORDS+1)
    from keras.models import Model
    from keras.layers import Input, Dense, Activation, Embedding, LSTM
    
    x_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='input')  #x_input.shape= (batch_size, 10)
        
    embed = Embedding(num_words,
                      5, 
                      embeddings_initializer='uniform',
                      input_length=MAX_SEQUENCE_LENGTH,
                      trainable=True,
                      mask_zero=True)

    h = embed(x_input)
    
    h = LSTM(5,return_sequences=False)(h)
    
    h = Dense(3,activation='softmax')(h) 
     
    
    model = Model(inputs = x_input, outputs = h)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model
model= build_model()
 
    

model.fit(data, labels, batch_size=2, epochs=10)
y_pred = model.predict(data)
print(data)
print(labels)
print(y_pred)



