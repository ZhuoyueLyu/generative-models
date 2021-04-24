import os
import pickle 
from MIDI_utils import parse_score
from music21 import instrument, converter

# Parameter 
save = True
load = True
note_len = 16 
folder_num = 10

dict_fname = "midi_data.bin"
if load:
    f = open(dict_fname, "rb")
    midi_dict = pickle.load(f)
    f.close() 
else:
    midi_dict = {
        'notes': [],
        'files': [], 
        'last_folder': 0, 
    }

path = "./clean_midi/" 
n_folders = len(os.listdir(path))

for cur_folder_count, folder in enumerate(os.listdir(path)[midi_dict['last_folder']:]):
    folder_left = folder_num - cur_folder_count
    if folder_left == 0:
        break 
    print("{} folders left".format(folder_num - cur_folder_count))
    for file in os.listdir(path+folder+"/"):
        fname = path+folder+"/"+file
        
        if file not in midi_dict['files']: 
            sc = instrument.partitionByInstrument(converter.parse(fname))
            if sc is None:
                continue 
            notes = parse_score(sc) 
            midi_dict['files'].append(file)
            for i in range(len(notes) // note_len):
                midi_dict['notes'].append(notes[i*note_len: (i+1)*note_len])

midi_dict['last_folder'] += folder_num 
print("Last folder: {}. Total folders: {}".format(midi_dict['last_folder'], len(os.listdir(path))))

if save:
    f = open(dict_fname, "wb")
    pickle.dump(midi_dict, f)
    f.close()
