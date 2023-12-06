#!/usr/bin/env python
# coding: utf-8

# # Generating Text with Neural Networks
# 

# # Getting the Data

# In[2]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()


# This step first block of code. is importing and downloading the file "shakespeare.txt" from the website provided, using tensorflow. Keras has downlaoded and named the file location as "filepath."

# In[3]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# This line of code is telling the program to print the first 80 characters of the shakespeare.txt file which was just downloaded.

# # Preparing the Data

# In[4]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]


# This code makes a text vectorisation layer and implements it to the shakespeare text. This line of code is mapping the words into vector real numbers.
# 
# 

# In[5]:


print(text_vec_layer([shakespeare_text]))


# This is telling the program to print (show)the text vectorisation layer that was just created.

# In[6]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394


# This line of code is calculating the number of disticnt characaters and the total number of characters that are present in the text file

# In[7]:


print(n_tokens, dataset_size)


# Printing the values that have just been calculated.

# In[9]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# This is converting the data to something that can be made into a data set that tensorflow is compatible with, to train the model.

# In[10]:


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:100_000], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[100_000:106_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)


# Here, i had to change the values as the code took for too long for the machine to learn. Through colaboration with peers we decided to change the values from 1 million to 100,000 to ensure that the model was practical. It was taking arounf 7 hours when I kept the original values, so switching to a lesser value greatly helped as it only took 1 hour.

# # Building and Training the Model

# In[11]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# This is training the model to run for 10 epochs. Its being trained to generate text character by character, using the text vis layer from earlier in the code. 

# In[12]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# In[13]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# This code here is telling the model to pick the most predictable character, after being trained. The model succesfully picks the right letter.

# In[14]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[15]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# The "next_char" term is the next charactee in the sequence. The temperature code is the randomness in which the letter is chosen.

# In[16]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# This is telling the model to extend the text and make it longer.

# In[17]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# In[18]:


print(extend_text("To be or not to be", temperature=0.01))


# This telling the model to print the extended version of the text, and is selecting the temperature. The number is low here so the text makes sense.

# In[19]:


print(extend_text("To be or not to be", temperature=1))


# In[20]:


print(extend_text("To be or not to be", temperature=100))


# In[21]:


print(extend_text("To be or not to be", temperature=1000))


# For testing, I made the temperature to a very high value, as you can see this ruined the text generatedl. The lower the temperature value, the more accurate the text created is.

# In[22]:


print(extend_text("To be or not to be", temperature=0.0001))


# In[23]:


print(extend_text("To be or not to be", temperature=0.000001))


# In[ ]:




