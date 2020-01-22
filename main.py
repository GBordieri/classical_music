# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:40:42 2019

@author: Garnet
"""
import tensorflow as tf
tf.enable_eager_execution()

import mido
import numpy as np
from pathlib import Path
import os
import midimanipulation as midman


INPUT_SIZE = midman.INPUT_SIZE
BATCH_SIZE = 1
EPOCHS = 1

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    # Use GPU if available
    if tf.test.is_gpu_available():
      rnn = tf.keras.layers.CuDNNGRU
    else:
      import functools
      rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')
      
    model = tf.keras.Sequential([
                tf.keras.layers.Dense(vocab_size, 
                                      input_shape=(INPUT_SIZE,),
                                      batch_input_shape=(1,INPUT_SIZE),
                                      activation=tf.nn.relu),
#                tf.keras.layers.Embedding(vocab_size, embedding_dim, 
#                              batch_input_shape=[batch_size, None]),
                tf.keras.layers.Reshape((1,INPUT_SIZE)),
                rnn(rnn_units,
                    return_sequences=True,
                    recurrent_initializer='glorot_uniform',
                    stateful=True),
                tf.keras.layers.Dense(vocab_size)
            ])
    return model


def split_input_target(track):
    input_track = track[:-1]
    target_track = track[1:]
    return input_track, target_track


def loss(inputs, outputs):
  return tf.keras.losses.mean_squared_error(inputs, outputs)


def generate_data():
    pathlist = Path('').glob('**/*.mid')
    super_track = np.zeros([1,INPUT_SIZE])  # dummy track to concatenate to 
    for path in pathlist:
        print("path: ", path)
        mid = mido.MidiFile(str(path))
        merged_track = midman.get_merged_piano_tracks(mid)
        merged_track = midman.track_to_tensor(merged_track)
        super_track = np.concatenate((super_track, merged_track), axis=0)
    super_track = np.delete(super_track, 0, 0)    # remove the blank message at the begining
    return super_track


def generate_midi(model):
    in_tensor = np.array([[0,0,1,0, 0, 0,0,0,0,0]])
    result = in_tensor
    model.reset_states()
    fp = open("output.txt", "w")
    fp.write(str(in_tensor) + '\n')
    
    max_msgs = 10000
    i = 0
    
    while np.any(in_tensor != [1,0,0,0, 0, 0,0,0,0,0]) and i < max_msgs:
        in_tensor = model(np.reshape(in_tensor,(1,10)))[0,:,:]
        print("prediction: ", in_tensor.shape, in_tensor)
        print("evaluated: ", in_tensor.numpy())
        fp.write(str(in_tensor) + '\n')
        result = np.concatenate((result, in_tensor), axis=0)
        i += 1
    fp.close()
    return result
    

if __name__ == '__main__':
    print("Main Start")
    
    ##### Create the model #####
    print("Building Model")
    model = build_model(INPUT_SIZE, 256, 128, BATCH_SIZE)
    model.summary()
    model.build((None, INPUT_SIZE))
    
    model.compile(
            optimizer = tf.train.AdamOptimizer(),
            loss = loss)
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    # Check if there are checkpoints available
    if os.path.isdir(checkpoint_dir) \
        and len(os.listdir(checkpoint_dir)) != 0:
        # If checkpoints are available, use them
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    else:
        
        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)
        
        # Generate the dataset
        print("Generating Dataset")
        super_track = generate_data()
        dataset = tf.data.Dataset.from_tensor_slices(super_track)
        dataset = dataset.batch(BATCH_SIZE+1, drop_remainder=True)
        dataset = dataset.map(split_input_target)
        dataset = dataset.repeat()
        
        #Train the model
        examples_per_epoch = len(super_track)//BATCH_SIZE
        steps_per_epoch = examples_per_epoch//BATCH_SIZE
        print("Training Start")
        print(dataset)
        history = model.fit(dataset,
                            epochs=EPOCHS,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=[checkpoint_callback])
    
    # Generate midi
    print("Generating Midi")
    result = generate_midi(model)
    midman.tensor_to_midi(result, 10000, "output")
    
    
    
    
      

        