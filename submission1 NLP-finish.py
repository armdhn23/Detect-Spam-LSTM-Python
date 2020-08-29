import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

df = pd.read_csv('spam.csv')
df = df[['type', 'text']]

df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')

category = pd.get_dummies(df.type)
df_baru = pd.concat([df, category], axis=1)
df_baru = df_baru.drop(columns='type')

text = df_baru['text'].values
label = df_baru[['ham', 'spam']].values

from sklearn.model_selection import train_test_split
text_latih, text_test, label_latih, label_test = train_test_split(text, label, test_size=0.2)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
     
tokenizer = Tokenizer(num_words=1000, oov_token='OOV')
tokenizer.fit_on_texts(text_latih) 
tokenizer.fit_on_texts(text_test)
     
sekuens_latih = tokenizer.texts_to_sequences(text_latih)
sekuens_test = tokenizer.texts_to_sequences(text_test)
     
padded_latih = pad_sequences(sekuens_latih) 
padded_test = pad_sequences(sekuens_test)

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 30

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\nAkurasi telah mencapai >90%!")
            self.model.stop_training = True
callbacks = myCallback()


history = model.fit(padded_latih, label_latih, epochs=num_epochs, callbacks=[callbacks],
                    validation_data=(padded_test, label_test), verbose=2)
                    
                    
y_pred = model.predict_classes(padded_test)
label = tf.argmax(label_test, axis = 1)
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix")
print(confusion_matrix(label, y_pred))
print("")
print("\t\t\tClassification Reports")
print(classification_report(label, y_pred))

print('\t\tDetect Spam or Ham')
print('##################')
more = "yes"
while(more=="yes"):
    new_text = [input('Enter Message: ')]
    df1 = pd.DataFrame(new_text) 
    
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    def clean_text(text):
        """
            text: a string
        
            return: modified initial string
        """
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
        text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
        return text
    df1 = df1[0].apply(clean_text)
    df1 = df1.str.replace('\d+', '')
    new_tokenizer = Tokenizer(num_words=1000, oov_token='OOV')
    new_tokenizer.fit_on_texts(new_text) 
    new_sekuens = tokenizer.texts_to_sequences(new_text)
    new_padded = pad_sequences(new_sekuens)
    result = model.predict(new_padded)
    if result[0,0] == result.max():
        result = "Ham"
        print('The message type is {}'.format(result))
        print('')
    else:
        result = "Spam"
        print('The message type is {}\n'.format(result))
        print('')
    more = input('Try another message? (yes/no) ').lower()
    if(more!="yes"):
        break
