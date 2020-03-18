# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:07:18 2020

@author: tasty_000
"""
from music21 import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def showScore(score):
    lpc = lily.translate.LilypondConverter()
    lpc.loadObjectFromScore(score)
    test = lpc.createPNG()
    img = mpimg.imread(str(test))
    plt.axis('off')
    plt.imshow(img)

#score = converter.parse('C:/Users/tasty_000/Documents/Jazzbot/Datasets/Test Data/midi_export.mxl')
works = corpus.getComposer('bach')

melodyWorks = []
harmonyWorks = []

#for work in works:
def byBarInput(input = None):
    
    if (input == None):
        works = corpus.getComposer('bach')
    
    melodyWorks = []
    harmonyWorks = []
    
    for work in works:
        if (len(work.parts) > 2):
            score = corpus.parse(work)
            length = (len(score.parts[0].getElementsByClass('Measure')))
            
            print("Name: " + work.name)
            print("Parts: " + str(len(score.parts)))
            print("Bars: " + str(length))
            
            melodyBars = []
            harmonyBars = []
            harmonyLabels = []
            
            #Grab individual bars
            for i in range(0, length-1):
                #print("Bar " + str(i+1))
                melody = []
                for note in score.measure(i).parts[0].pitches:
                    melody.append(note.name)
                melodyBars.append(melody)
                
                harmony = []
                for j in range(0, (len(score.parts))):
                    pitches = score.measure(i).parts[j].pitches
                    if (len(pitches) != 0):
                        harmony.append(pitches[0].name)
                harmonyBars.append(harmony)
            melodyWorks.append(melodyBars)
            harmonyWorks.append(harmonyBars)       
            
    return [melodyWorks, harmonyWorks]    
    
print("Harmony: ")    
print(harmonyWorks)
print("Melody: ")
print(melodyWorks)
            
