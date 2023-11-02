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


# In[19]:


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

# In[27]:


from matplotlib import pyplot


# In[28]:


test_picture = pyplot.imread("picture1.jpg")
print("Nump array of the image is: ",test_picture)
pyplot.imshow(test_picture)


# 
