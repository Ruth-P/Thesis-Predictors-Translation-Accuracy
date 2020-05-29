#!/usr/bin/env python
# coding: utf-8

# Input Duolingo datafiles from (Settles, Burr, 2018, "Data for the 2018 Duolingo Shared Task on Second Language 
## Acquisition Modeling (SLAM)", https://doi.org/10.7910/DVN/8SWHNO, Harvard Dataverse, V4)
## Brysbaert Concreteness scores, Subtlex_En and Subtlex_Esp from:
# (Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand  
# generally known English word lemmas. Behavior Research Methods, 46, 991-997.)
# (Brysbaert, M., Mandera, P., & Keuleers, E. (2018). The Word Frequency Effect in Word Processing: 
# An Updated Review. Current Directions in Psychological Science, 27(1), 45–50.)
#(Cuetos, F., Glez-Nosti, M., Barbón, A., & Brysbaert, M. (2011). SUBTLEX-ESP: Spanish word 
# frequencies based on film subtitles. Psicológica, 32(2), 133–143.)


import gzip
import os 
import tarfile
import math
import argparse
import xlrd
import csv
import numpy as np
import pandas as pd
import Levenshtein as lev
import seaborn as sns
from collections import defaultdict, namedtuple
from random import shuffle, uniform
from io import open
from future.builtins import range
from future.utils import iteritems
from sklearn import preprocessing

## Function to Load Data and Extract Features: Returns 3D Array with : User, Format, Token, PoS and Mistake(0 or 1) per Instance. 
def load_data(filename):
    data = []
    instance_properties = dict() 
    with open(filename, 'rt', encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()           
            if len(line) == 0:                              # empty line means end of exercise
                instance_properties = dict()
            elif line[0] == '#':                             # line starts with #, means new exercise
                if 'prompt' in line:
                    pass
                else:
                    list_of_exercise_parameters = line[2:].split()
                    for exercise_parameter in list_of_exercise_parameters:
                        [key, value] = exercise_parameter.split(':')
                        if key == 'user':
                            instance_properties[key] = value
                        elif  key == 'format':
                            instance_properties[key] = value   
            else:                            ## new Instance for the current exercise
                line = line.split()             
                instance_properties['token'] = str.lower(line[1])
                instance_properties['part_of_speech'] = line[2]                      
                instance_properties['mistake'] = float(line[6])
                data.append(list(instance_properties.items()))       
        return np.asarray(data)

## Load data and extract features per track
en_es = load_data("en_es.slam.20190204.train")
es_en = load_data("es_en.slam.20190204.train")
fr_en = load_data("fr_en.slam.20190204.train")

## Create 2 dictionaries: 1) Number of Mistakes per User, per Word, per Format (Nr_Mistakes) 
## 2) Number of times word was seen, per user per Format (Word_exp) . 

def mistake_wordexp_dct(data):
    uwf_mistakes=defaultdict(int)
    word_exp=defaultdict(int)
    
    for i in range(data.shape[0]):
        uwf_mistakes[(data[i][0][1]),(data[i][2][1]), (data[i][1][1]), (data[i][3][1])] += int(float((data[i][-1][1])))
        word_exp[(data[i][0][1]),(data[i][2][1]), (data[i][1][1]), (data[i][3][1])] +=1
    return uwf_mistakes, word_exp 
    
## Create a dictionary that maps English words to Concreteness and Frequency score
Brysbaert=pd.read_excel('BrysbaertConcretenessFrequency.xlsx')
Brysbaert=np.array(Brysbaert)
En_freq=pd.read_excel('SUBTLEX_En.xlsx')
En_freq=np.array(En_freq)

def dct_Brys_Enfreq(brysbaert, en_freq):
    dct_concr={}
    dct_freq={}
    for line in brysbaert:
        dct_concr[line[0]]=line[2]   ## first concreteness,
    for line in en_freq:
        dct_freq[line[0]] = line[5]    
    return dct_concr, dct_freq

## Create Spanish frequency dictionary
Spanish_freq=pd.read_excel('SUBTLEX-ESP.xlsx')
Spanish_freq=np.array(Spanish_freq)

def dct_Spanish(Spanish_freq):
    dct={}
    for line in Spanish_freq:
        dct[line[0]] = line[2]
        dct[line[5]] = line[7]
        dct[line[10]] = line[12]
    return dct

## Function to add ftrs Word_exp, Language track, Word length and Nr_Mistakes. 
def ftrs_2(dct_mistakes, dct_exp, track):
    extd_ftrs=np.empty((len(dct_mistakes), 12), dtype=object)        ## empty array to contain all features. 
    mist_keys=np.asarray(list(dct_mistakes))
    exp_keys=np.asarray(list(dct_exp))
    mist_values=np.asarray(list(dct_mistakes.values()))
    exp_values=np.asarray(list(dct_exp.values()))

    for i in range(len(extd_ftrs)):
        extd_ftrs[i,0:4]=(mist_keys[i])
        extd_ftrs[i,-1]=mist_values[i]
        extd_ftrs[i,5]=track
        extd_ftrs[i,6]=exp_values[i]
    for j in extd_ftrs:
        j[4]=len(str(j[1]))       ## word length 
    return extd_ftrs

## Define dictionaries 
dct_mistakes_en_es, dct_exp_en_es = mistake_wordexp_dct(en_es)
dct_mistakes_es_en, dct_exp_es_en = mistake_wordexp_dct(es_en)
dct_mistakes_fr_en, dct_exp_fr_en = mistake_wordexp_dct(fr_en)

dct_Brysbaert, dct_Enfreq=dct_Brys_Enfreq(Brysbaert, En_freq)
dct_Spanish_freq=dct_Spanish(Spanish_freq)

## Extend ftrs per track (Word_exp, Word_length, Track, and Nr_Mistakes)
en_es_2 = ftrs_2(dct_mistakes_en_es, dct_exp_en_es, 'en_es')
es_en_2 = ftrs_2(dct_mistakes_es_en, dct_exp_es_en, 'es_en')
fr_en_2 = ftrs_2(dct_mistakes_fr_en, dct_exp_fr_en, 'fr_en')

## Load the excel files with translation
en_es_trans=pd.read_excel('en_sp_words.xlsx')
en_es_trans=np.array(en_es_trans)

es_en_trans=pd.read_excel('es_en_words.xlsx')
es_en_trans=np.array(es_en_trans)

fr_en_trans=pd.read_excel('fr_en_words.xlsx')
fr_en_trans=np.array(fr_en_trans)

## Function for creating Translation Dictionary
def trans_dct(trans_array):
    dct = dict()
    for i in trans_array:
        dct[i[0]] = i[1]
    return dct

## Define translation dictionaries for each track
en_es_dct = trans_dct(en_es_trans)
es_en_dct = trans_dct(es_en_trans)
fr_en_dct = trans_dct(fr_en_trans)

## Function for adding Concreteness, Frequency and Distance to ftr vectors
def ftrs_3(data, trans_dct, track): ## remove dct_brys and dct_sp_freq from input, these are constant    
    for instance in data:
        if instance[1] in trans_dct:
            L2_word = instance[1]
            L1_word = str(trans_dct[L2_word]).lower()         
            if track == "en_es":                                ## NB: es = L1, en = L2, Spanish learning English. 
                if L1_word in dct_Spanish_freq and L2_word in dct_Brysbaert:
                    dis=lev.distance((L1_word), str(L2_word))
                    instance[7]=dct_Spanish_freq[L1_word]
                    instance[8]=dis
                    instance[9]=dct_Brysbaert[L2_word]
    
            if track == "es_en":
                if instance[3] == 'VERB' and len(L1_word.split()) > 1:
                    L1_word = L1_word.split()[-1]                  ## spanish verbs are often translated into english with pronouns, this takes only the verb (the last word) 

                if L1_word in dct_Brysbaert and L1_word in dct_Enfreq:
                    dis=lev.distance((L1_word), str(L2_word))         
                    instance[7]=dct_Enfreq[L1_word]
                    instance[8]=dis
                    instance[9]=dct_Brysbaert[L1_word]
                    
            if track == "fr_en":            
                if L1_word in dct_Brysbaert and L1_word in dct_Enfreq:
                    dis=lev.distance((L1_word), str(L2_word))          
                    instance[7]=dct_Enfreq[L1_word]
                    instance[8]=dis
                    instance[9]=dct_Brysbaert[L1_word]
    return data

## Add Concreteness, Frequency and Distance to ftr vectors
en_es_3 = ftrs_3(en_es_2, en_es_dct, 'en_es')
es_en_3 = ftrs_3(es_en_2, es_en_dct, 'es_en')
fr_en_3 = ftrs_3(fr_en_2, fr_en_dct, 'fr_en')

## Unfamiliar Sound Spellings
En_Es_Unf = ['ee', 'augh', 'tt', 'ow', 'oa', 'oo', 'ou']
Fr_En_Unf = ['ui', 'eu', 'un', 'in', 'ai', 'en', 'ei', 'oeu']
Es_En_Unf = ['ll', 'j', 'x']

## Dictionary with Word as Key, and Unfamiliar sound (1/0) as value
## Include unfamiliar sound ftr in final ftr vectors
def ftr_unf_sound(data, unique_words, unf_sound_list):
    dct_ftr = dict()
    
    for word in unique_words:
        unf_sound = []
        for spelling in unf_sound_list:
            unf_sound.append(spelling in word)
            if any(unf_sound):
                dct_ftr[word] = 1
            else:
                dct_ftr[word] = 0
    for instance in data:
        instance[10] = dct_ftr[instance[1]]   
    return dct_ftr, data

## Define dictionaries, and complete ftrs 
en_es_dct_unf, en_es_compl = ftr_unf_sound(en_es_3, unique_words_en, En_Es_Unf)
es_en_dct_unf, es_en_compl = ftr_unf_sound(es_en_3, unique_words_es, Es_En_Unf)
fr_en_dct_unf, fr_en_compl = ftr_unf_sound(fr_en_3, unique_words_fr, Fr_En_Unf)

## Concatenate all tracks
ftrs_compl = np.concatenate((en_es_compl, es_en_compl, fr_en_compl), axis = 0)

## Create dictionary with number of observation per user. 
def user_counts(data):
    user_dct = dict()
    users_lst = list(data[:,0])
    unique_users = (set(users_lst))
    for user in unique_users:
        user_dct[user] = users_lst.count(user) 
    return user_dct

user_counts = user_counts(ftrs_compl)

## Function to mark users outside of 10th and 90th percentile
def drop_users(user_counts, data):    
    drop_users = []
    dropped_instances = 0
    
    for user in user_counts.keys():
        if user_counts[user]<200 or user_counts[user]>630:
            drop_users.append(user)
            
    for row in data:
        if row[0] in drop_users:
            row[0] = None
            dropped_instances +=1
    return drop_users, data, dropped_instances

drop_users, ftrs_compl, dropped_instances = drop_users(user_counts, ftrs_compl)

## Turn into dataframe, drop words that have no frequency and/or concreteness scores, and marked users
df_ftrs=pd.DataFrame(ftrs_compl)
df_ftrs=df_ftrs.dropna()
df_ftrs.columns=['User', 'Word', 'Task_Format', 'PoS', 'Word_length', 'Track', 'Word_exp', 'Frequency', 'Distance', 'Concreteness', 'Unfamiliar_Sound', 'Nr_Mistakes']
df_ftrs.to_csv("ftrs.csv")

