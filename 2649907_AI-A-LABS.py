#!/usr/bin/env python
# coding: utf-8

# # Week 1 - :  Getting started with Anaconda, Jupyter Notebook and Python
# 
# I chose AI because I am interested in its potential capabilities in the near future
# 
# I have no experience with AI or python, however I find coding to be easy
# 
# 
# 
# 

# Link to my github repository - https://github.com/BorthwickJ/BorthwickJ.github.io.git

# In[1]:


print ("Hello World!") 


# In[2]:


message = "Hello World!"

print (message)


# In[3]:


greeting = "Yo wat up"

print (greeting)


# In[4]:


print (message + greeting)


# In[5]:


print (message*3)


# In[6]:


print (message[0])


# In[7]:


print (message[2])


# 
# •	What happens when you print message + message? What is your output? - it puts both the message and the greeting together.
# 
# •	What happens when you print message*3? What is your output? - The messsge is shown 3 times
# 
# •	What happens when you print message [0]? What is your output? Why? What if you change 0 to a different number? - it just prints the first letter. This is because the number zero is matched with the letter H. The number 2 is matched with the letter L
# 
# •	Do you think message is a good variable name? - i think its is a good variable name for basic coding as it does what it says - prints a message
# 

# In[8]:


from IPython.display import *


# In[9]:


YouTubeVideo("KnumAWWWgUE")


# In[10]:


import webbrowser
import requests



print("Shall we hunt down an old website?")
site = input("Type a website URL: ")
era = input("Type year, month, and date, e.g., 20150613: ")
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era)
response = requests.get(url)
data = response.json()
try:
    old_site = data["archived_snapshots"]["closest"]["url"]
    print("Found this copy: ", old_site)
    print("It should appear in your browser.")
    webbrowser.open(old_site)
except:
    print("Sorry, could not find the site.")



# # Week 2 - exploring data in multiple ways 
# 
# 

# In[11]:


from IPython.display import Image 


# In[12]:


Image ("picture1.jpg")


# In[13]:


from IPython.display import Audio


# In[14]:


Audio ("audio1.mid")


# In[15]:


Audio ("GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg") #This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license.
#You are free: 
#•	to share – to copy, distribute and transmit the work
#•	to remix – to adapt the work
#Under the following conditions: 
#•	attribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
#•	share alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
#The original ogg file was found at the url: 
#https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# Only the second audio played. I think the second Audio Played because it is an embedded html audio. The first file also cannot be opened on my laptop, as there is no actual audio on it. 

# # LIBRARY MATPLOTLIB

# In[16]:


from matplotlib import pyplot


# In[17]:


test_picture = pyplot.imread("picture1.jpg")
print("Nump array of the image is: ",test_picture)
pyplot.imshow(test_picture)


# 

# In[18]:


from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    


# In[19]:


from sklearn import datasets


# In[20]:


dir(datasets)


# I chose the datasets load_sample_images and load_wine as i was curious what would happen

# In[21]:


wine_data = datasets.load_wine()


# In[22]:


SampleImages_data = datasets.load_sample_images()


# In[23]:


wine_data.DESCR


# In[24]:


SampleImages_data.DESCR


# In[25]:


wine_data.feature_names


# Wine datasets has 13 feature within it.

# I noticed when trying to use sample images, the program refused to present the names. perhaos because it is image properties and doesnt have many features.

# In[26]:


from sklearn import datasets
import pandas

wine_data = datasets.load_wine()

wine_dataframe = pandas.DataFrame(data=wine_data["data"], columns = wine_data["feature_names"])


# I think that the pandas.datafram comman changes the names of what the things mean. For examoke it makes the data bleong to the winde data, the columns to wine data.

# # WEEK 5 - Deep Dive Machine Learning Group Task

# ![alt text](images/Diagram.jpg)
# 
# The example we used in the lecture was to do with leukaemia. The data is the patients that have leaukaemia. The algorith for this example was "Patient name = Luke, if yes = suspect leukaemia if no = all clear. The output is the decision that it makes. 
# 
# For a museum curator, you could create a diagram, for example a flowchart, of sorting different parts of a collection. For example, "If art from France yes = curate into french collection if no = do not curate into french collection. 
# 
# Flowcharts are an excellent way of describing to a wider audience as they are easy to follow. 
# 
# An example we also used was in lecture 4. It was images of shells on a beach. The algorith is a neaural network. It is trained to match patterns to either a starfish or a sea urchin. The output of the model matches pattersn found on a beach to the recognised elements of the starfish or sea urchin, for example pattern, shape. It would need to be supervised to ensure there are no false positives. If it is positive it can be fed back into the network for next time use. 
# 
# We also looked at a spam filtering within neural networking. It can be used to detect and organise spam emails. The algorith is neural networks. The alogiryth gives weights to randomised words as a starting point. The accuracy of the model can be checked by humans. This alters the weight of the spam words, the more likely the words used are spam, the heavier the weight that is given to the words. 
# 
# You can evaluate the performance of a model is using cross validation. The data is split into test data training data. The performance is the same on both test and training data. 

# # - WEEK 5 TASK - MACHINE LEARNING BY EXAMPLE 1-4 Framing the problem

# I believe that regression would be best for prediciting median price ranges as there are many different factors that apply such as, size, area, rooms. Linear regression would allow for for the median to be calculated based on these factors. As prior mentioned the size of house, area, rooms all play a part in the prediction for the median price. Cultural differences could play apart in this bias as places that may be seen as lesser to some, will be better for others. Income differences also play a part into this.
# 
# For handwritten recognition, classification would be best as the writing input is being defined as a discrete set of letters.
# 
# 

# # - WEEK 5 TASK 2

# In[27]:


from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data(): #defines a function that loads the housing data available as .tgz file on a github URL
    tarball_path = Path("datasets/housing.tgz") # where you will save your compressed data
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True) #create datasets folder if it does not exist
        url = "https://github.com/ageron/data/raw/main/housing.tgz" # url of where you are getting your data from
        urllib.request.urlretrieve(url, tarball_path) # gets the url content and saves it at location specified by tarball_path
        with tarfile.open(tarball_path) as housing_tarball: # opens saved compressed file as housing_tarball
            housing_tarball.extractall(path="datasets") # extracts the compressed content to datasets folder
    return pd.read_csv(Path("datasets/housing/housing.csv")) #uses panadas to read the csv file from the extracted content

housing = load_housing_data() #runsthe function defined above


# In[28]:


housing.info()


# In[29]:


housing["ocean_proximity"].value_counts()


# In[30]:


housing.hist(bins=50, figsize=(12, 8))


# In[31]:


housing.describe()


# In[32]:


from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')


# In[33]:


print(mnist.DESCR)


# # Task 2-3: Review the data description above with your group.
# 

# The images were normalised to fit in a 20x20, but were centered into 28x28, allowing each pixel to be seen.
# 
# They combined and merged special database 1 with special database 3 and then retook samples into new test and training data sets. I think was justified as it is now essential to have both training datasets and test datasets to ensure that the results are not biased. It also allows for testing to see if the final model is working correctly. Finally, its stops the model from overfitting. This is when the model learns the training data too well and is unable to generalise new data.
# 
# 
# 
# 
# 
# 

# # Task 2-4

# In[34]:


mnist.keys()


# In[35]:


# cell for python code 

images = mnist.data
categories = mnist.target

# insert lines below to print the shape of images and to print the categories.
print(images.shape)
list(categories)


# In[36]:


#extra code to visualise the image of digits

import matplotlib.pyplot as plt

## the code below defines a function plot_digit. The initial key work `def` stands for define, followed by function name.
## the function take one argument image_data in a parenthesis. This is followed by a colon. 
## Each line below that will be executed when the function is used. 
## This cell only defines the function. The next cell uses the function.

def plot_digit(image_data): # defines a function so that you need not type all the lines below everytime you view an image
    image = image_data.reshape(28, 28) #reshapes the data into a 28 x 28 image - before it was a string of 784 numbers
    plt.imshow(image, cmap="binary") # show the image in black and white - binary.
    plt.axis("off") # ensures no x and y axes are displayed


# In[37]:


some_digit = mnist.data[0]
plot_digit(some_digit)
plt.show()


# # TASK 3 - SETTING ASIDE DATA

# In[38]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

train_set, test_set = train_test_split(housing, test_size=tratio, random_state=42) 
## assigning a number to random_state means that everytime you run this you get the same split, unless you change the data.


# In[39]:


# extra code – shows another way to estimate the probability of bad sample

import numpy as np

sample_size = 1000
ratio_female = 0.511

np.random.seed(42)

samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
((samples < 485) | (samples > 535)).mean()


# In[40]:


import numpy as np
import pandas as pd

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[41]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

strat_train_set, strat_test_set = train_test_split(housing, test_size=tratio, stratify=housing["income_cat"], random_state=42)


# In[42]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set) #Prints out in order of the highest proportion first.


# It is reasonable to use a stratisifed sample based on median income as it allows for diverse subgroups to be included within their samples. It helps to provide estimates of characteristics belonging to each group. Makes sure the data taken in isnt biased and allows for a vast group of people to look at it.

# In[43]:


type(mnist.data)


# # TASK 3.2 - SETTING ASIDE DATA

# In[44]:


X_train = mnist.data[:60000]
y_train = mnist.target[:60000]

X_test = mnist.data[60000:]
y_test = mnist.target[60000:]


# # TASK 4 

# In[45]:


housing = strat_train_set.copy()


# In[46]:


corr_matrix = housing.corr(numeric_only=True) # argument is so that it only calculates for numeric value features
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[47]:


from pandas.plotting import scatter_matrix

features = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[features], figsize=(12, 8))
#save_fig("scatter_matrix_plot")  

#The line above is extra code you can uncomment (remove the hash at the begining) to save the image.
#But, to use this, make sure you ran the code at the beginning of this notebook defining the save_fig function

plt.show()


# In[48]:


housing = strat_train_set.drop("median_house_value", axis=1) ## 1)
housing_labels = strat_train_set["median_house_value"].copy() ## 2)


# In[49]:


housing.info()


# There are 168 values missing for total_bedrooms

# In[50]:


# this is the code for Option 1 above. 
housing_option1 = housing.copy() #This makes a copy of the data to variable housing_option1, so that we don't mess up the original data.

housing_option1.dropna(subset=["total_bedrooms"], inplace=True)  # option 1 - dropping the rows where total_bedroom is missing values.

housing_option1.info() #look for missing values after rows have been dropped


# In[51]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # initialises the imputer

housing_num = housing.select_dtypes(include=[np.number]) ## includes only numeric features in the data

imputer.fit(housing_num) #calculates the median for each numeric feature so that the imputer can use them

housing_num[:] = imputer.transform(housing_num) # the imputer uses the median to fill the missing values and saves the result in variable X


# In[52]:


housing_num.describe()


# In[53]:


from sklearn.preprocessing import MinMaxScaler # get the MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # setup an instance of a scaler
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)# use the scaler to transform the data housing_num


# In[54]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


# In[55]:


housing_num[:]=std_scaler.fit_transform(housing_num)


# In[56]:


target_scaler = StandardScaler() #instance of Scaler
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame()) #calculate the mean and standard deviation and use it to transform the target labels.


# In[57]:


from sklearn.linear_model import LinearRegression #get the library from sklearn.linear model

model = LinearRegression() #get an instance of the untrained model
model.fit(housing_num, scaled_labels)
#model.fit(housing[["median_income"]], scaled_labels) #fit it to your data
#some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

#scaled_predictions = model.predict(some_new_data)
#predictions = target_scaler.inverse_transform(scaled_predictions)


# In[58]:


some_new_data = housing_num.iloc[:5] #pretend this is new data
#some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# In[59]:


print(predictions, housing_labels.iloc[:5])


# In[60]:


from sklearn.model_selection import cross_val_score

rmses = -cross_val_score(model, housing_num, scaled_labels,
                              scoring="neg_root_mean_squared_error", cv=10)


# In[61]:


pd.Series(rmses).describe()


# # TASK 4-2

# In[63]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data()


# In[64]:


print(type(mnist))


# In[65]:


(X_train_full, y_train_full), (X_test, y_test) = mnist 
# (X_train_full, y_train_full) is the 'tuple' related to `a` and (X_test, y_test) is the 'tuple' related to `b`.
# X_train_full is the full training data and y_train_full are the corresponding labels 
# - labels indicate what digit the image is of, for example 5 if it is an image of a handwritten 5.


# In[66]:


X_train_full = X_train_full / 255.
X_test = X_test / 255.


# In[67]:


X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# In[68]:


import numpy as np # you won't need to run this line if you ran it before in this notebook. But for completeness.

X_train = X_train[..., np.newaxis] #adds a dimension to the image training set - the three dots means keeping everything else the same.
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# In[69]:


tf.keras.backend.clear_session()

tf.random.set_seed(42)
np.random.seed(42)

# Unlike scikit-learn, with tensorflow and keras, the model is built by defining each layer of the neural network.
# Below, everytime tf.keras.layers is called it is building in another layer

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", 
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))


# In[70]:


model.evaluate(X_test, y_test)


# In[71]:


from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')


images = mnist.data
categories = mnist.target


# In[72]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd_clf = SGDClassifier(random_state=42)

#cross validation on training data for fit accuracy

accuracy = cross_val_score(sgd_clf, images, categories, cv=10)

print(accuracy)


# 
# - Your were to use your own data (for example, discuss survey data data and photos) - i would need to resize my images to 100x100 so that they would fit properly. 
#     
#     
#     - Your model? - for images i would use a Neural network as it is the best for images. for the survey data i would use a decision teee as personally i understand them the most.  
#     
#     - Your scaling method? for the images i would use the tensor flow teras, as it worked effectively in the example prior. I could utilise layer normalisation as my data would be smaller, so batch normalisation wouldnt work. 
#     
#     - Your approach to handling missing data? - an issue with my own data may be data privacy, as it is stored on my own device and could be stolen and mishandled. Id esnure to keep my data protected and not be stolen and used by others.
# 
# 
# Cross validating allows for confidence that the model will perform well. By comparing with another model it allows for the the model to learn from the training data.
# 
# 

# In practice, i couldnt actually get the training loss down to 0.2. i got stuck at around 0.34. i had 2000 epochs and it wasnt getting any lower. It had 3 hidden layers and 2 exta neutrons.  
# 
# We found it difficult to determine what patterns would be present at different layers. Personally, I found this hard to wrap my head around and interact with. Im sure if I kept trying I would get it, but overall I struggled.

# In[73]:


from keras.applications.vgg19 import VGG19

model = VGG19() ### this will take some time!!


# In[74]:


print(model.summary())


# # CRITICALLY ENGAGING WITH AI ETHICS TASK 2 

# There are 6 types of bias that are described on the first page. Before the course I knew of historical bias and measurement bias. I didnt know of representation, aggregation, eveluation and deployment. Other types of bias that ive heard of before is confirmation bias and gender bias. 
# 
# The types if bias present in the kaggle example which I found was, representation, evaluation, and measurement. attached are screenshots of where I found amn example of these. 
# 
# ![alt text](images/evaluation.jpg)
# 
# ![alt text](images/historical.jpg)
# 
# ![alt text](images/measurement.jpg)
# 
# 

# # TASK 3

# From my findings, words that were similar to eachother did indeed sit next to eachother. Words related to the apple sat next to eachother. Fruits grouped together and so were words to do with the apple brand such as marketing, logo and mackintosh. For silver all of the words that were generated were other metals. some were closer together than others, depending on the chemical makeup of the material.
# 
# When exploring other occupations, similar occupations came up. when drummer was searched, words like guitariust abd bassist appeared. I would argue that gender bias does not come into it, as nowadays lots of women are venturing into more male dominated fields. Perhaps, because of the worrds used it could come across as gendeer bias. When the word "police" is searched, policeman and men comes up but police woman doesnt. 
# 
# ![alt text](images/screenshot.jpg)
# 
# 
# 
# 
# 

# # TASK 4 

# There are 4 criteria that are explained on the kaggle tutorial page. The only criteria that I had heard before the course was equal opportunity. I had not heard equal accuracy, demographic parity and group unaware.
# 
# We couldnt think of any more criteria that would apply to AI fairness.
# 
# I found that the answers I gave whilst completing the tutorial to be correct. 

# ![alt text](images/task4-1.jpg)
# 
# ![alt text](images/task4-2.jpg)
# 
# ![alt text](images/task4-3.jpg)
# 
# ![alt text](images/task4-4.jpg)
# 
# ![alt text](images/task4-5.jpg)

# There were 8 features present within the dataset.
# 
# I found the code quite hard to do and had to eventually utilise the hint and answer to see where I went wrong. I struggled with the show_weights step. eventually I got it to work. I managed to use my knowledge from the first example to help the second which I managed to do fine.
# 
# attached are screenshots of the tasks i completed
# 
# 
# 
# ![alt text](images/task5-2.jpg)
# 
# ![alt text](images/task5-3.jpg)
# 
# ![alt text](images/task5-4.jpg)
# 
# ![alt text](images/task5-5.jpg)
# 
# ![alt text](images/task5-6.jpg)
# 
# ![alt text](images/task5-7.jpg)
# 
# ![alt text](images/task5-8.jpg)
# 
# 

# In[ ]:





# In[ ]:




