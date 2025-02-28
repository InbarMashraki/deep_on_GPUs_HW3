# CS236781: Deep Learning on Computational Accelerators
# Homework Assignment 3

Faculty of Computer Science, Technion.

## Introduction

In this assignment we'll learn to generate text with a deep multilayer RNN network based on GRU cells.
Then we'll focus our attention on image generation using a variational autoencoder.
We will then shift our focus to sentiment analysis: First by training a transformer-style encoder, and then by fine-tuning a pre-trained model from Hugging Face.


## General Guidelines

- Please read the [getting started page](https://vistalab-technion.github.io/cs236781/assignments/getting-started) on the course website. It explains how to **setup, run and submit** the assignment.
- This assignment requires running on GPU-enabled hardware. Please read the [course servers usage guide](https://vistalab-technion.github.io/cs236781/assignments/hpc-servers). It explains how to use and run your code on the course servers to benefit from training with GPUs.
- The text and code cells in these notebooks are intended to guide you through the
  assignment and help you verify your solutions.
  The notebooks **do not need to be edited** at all (unless you wish to play around).
  The only exception is to fill your name(s) in the above cell before submission.
  Please do not remove sections or change the order of any cells.
- All your code (and even answers to questions) should be written in the files
  within the python package corresponding the assignment number (`hw1`, `hw2`, etc).
  You can of course use any editor or IDE to work on these files.

## Contents
- [Part1: Sequence Models](#part1)
    - [Text generation with a char-level RNN](#part1_1)
    - [Obtaining the corpus](#part1_2)
    - [Data Preprocessing](#part1_3)
    - [Dataset Creation](#part1_4)
    - [Model Implementation](#part1_5)
    - [Generating text by sampling](#part1_6)
    - [Training](#part1_7)
    - [Generating a work of art](#part1_8)
    - [Questions](#part1_9)
- [Part 2: Generative adverserial network](#part2)
- [Part 3: Transformer Encoder](#part3)
    - [Reminder: scaled dot product attention](#part3_1)
    - [Sliding window attention](#part3_2)
    - [Multihead Sliding window attention](#part3_3)
    - [Sentiment analysis](#part3_4)
    - [Obtaining the dataset](#part3_5)
    - [Tokenizer](#part3_6)
    - [Transformer Encoder](#part3_7)
    - [Training](#part3_8)
    - [Questions](#part3_9)
- [Part 4: Fine-tuning a pretrained language model](#part4)
    - [Loading the dataset](#part4_1)
    - [Tokenizer](#part4_2)
    - [Loading pre-trained model](#part4_3)
    - [Fine-tuning](#part4_4)
    - [Questions](#part4_5)
# deep_on_GPUs_HW3
