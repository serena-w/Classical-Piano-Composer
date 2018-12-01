""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, tempo, key, meter
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Activation, Concatenate
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    (notes,lengths) = get_notes()

    # get amount of pitch names
    n_notes = len(set(notes))
    n_lengths = len(set(lengths))

    network_input, notes_output, lengths_output = prepare_sequences(notes, lengths, n_notes,
        n_lengths)

    model = create_network(network_input, n_notes, n_lengths)

    train(model, network_input, notes_output, lengths_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    lengths = []
    first = True

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for (i,element) in enumerate(notes_to_parse):
            if isinstance(element, note.Note):
                notes.append("note:" + str(element.pitch))
                lengths.append(str(element.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append("chord:" + ('.'.join(str(n) for n in
                  element.normalOrder)))
                lengths.append(str(element.quarterLength))
            elif isinstance(element, tempo.MetronomeMark):
                if i < 5:
                    notes.append("metro:" +
                      str(int(5 * round(float(element.number)/5))))
                    lengths.append("")
            elif isinstance(element, note.Rest):
                notes.append("rest:" + str(element.quarterLength))
                lengths.append("")
            """
            elif isinstance(element, instrument.Piano):
                notes.append("piano")
            elif isinstance(element, key.Key):
                notes.append("key:" + element.tonicPitchNameWithCase)
            elif isinstance(element, meter.TimeSignature):
                notes.append("time:" + element.ratioString)
            else:
              print("not accounted for",element,element.duration)
            """

        notes.append('END')
        lengths.append('END')
        if first:
          print(notes)
          first = False

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open('data/lengths', 'wb') as filepath:
        pickle.dump(lengths, filepath)

    return (notes,lengths)

def prepare_sequences(notes, lengths, n_notes, n_lengths):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100
    print("notes length",len(notes))
    print("lengths length",len(lengths))

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # get all lengths
    valid_lengths = sorted(set(item for item in lengths))

    print("pitchnames length",len(pitchnames))
    print("valid_lengths length",len(valid_lengths))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

     # create a dictionary to map lengths to integers
    length_to_int = dict((length, number) for number, length in
        enumerate(valid_lengths))

    notes_input = []
    lengths_input = []
    notes_output = []
    lengths_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        notes_in = notes[i:i + sequence_length]
        lengths_in = lengths[i:i + sequence_length]
        notes_out = notes[i + sequence_length]
        lengths_out = lengths[i + sequence_length]
        notes_input.append([note_to_int[char] for char in notes_in])
        lengths_input.append([length_to_int[l] for l in lengths_in])
        notes_output.append(note_to_int[notes_out])
        lengths_output.append(length_to_int[lengths_out])

    # normalize input
    notes_input = np.array(notes_input) / float(n_notes)
    lengths_input = np.array(lengths_input) / float(n_lengths)
    print("notes_input shape", notes_input.shape)
    print("lengths_input shape", lengths_input.shape)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.stack([notes_input,lengths_input],axis=-1)
    print("network_input shape", network_input.shape)

    notes_output = np_utils.to_categorical(notes_output)
    lengths_output = np_utils.to_categorical(lengths_output)
    print("notes_output shape", notes_output.shape)
    print("lengths_output shape", lengths_output.shape)

    return (network_input, notes_output, lengths_output)

def create_network(network_input, n_notes, n_lengths):
    """ create the structure of the neural network """
    inputs = Input(shape=(network_input.shape[1], network_input.shape[2]))
    lstm1 = LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )(inputs)

    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(512, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    lstm3 = LSTM(512)(dropout2)
    dense1 = Dense(256)(lstm3)
    dropout3 = Dropout(0.3)(dense1)

    notes_layer = Dense(n_notes)(dropout3)
    notes_output = Activation('softmax', name="notes_output")(notes_layer)
    lengths_layer = Dense(n_lengths)(dropout3)
    lengths_output = Activation('softmax', name="lengths_output")(lengths_layer)
    # output_layer = Concatenate()([notes_output,lengths_output])

    model = Model(inputs=inputs, outputs=[notes_output,lengths_output])
    model.summary()

    losses = {
      "notes_output": "categorical_crossentropy",
      "lengths_output": "categorical_crossentropy",
    }
    lossWeights = {"notes_output": 1.0, "lengths_output": 1.0}
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, notes_output, lengths_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, {"notes_output": notes_output, "lengths_output":
      lengths_output}, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
