# An Introduction to Transformers in Biology

Problem: 
Given an amino acid sequence, find a way to reliably predict the corresponding secondary structure sequence. 

Initial Thoughts:

Token Classification
The first thing that came to mind was labeling this problem as a sequence to sequence (seq2seq) problem. Afterall, botht the input and the output are sequence so it is logical to assume that secondary structure prediction is a seq2seq problem. In reality, the goal is to label each of the input amino acid sequences as a part of a certain secondary structure. This entails that the output sequence be the same length as the input sequence. In other words, a one to one coresspondence. The task of labeling or classifying characters in an input sequence as part of a specfic group is called token classification. Each token, or amino acid residue, will be classified into a destinct category which represents a secondary structure. 
