import pickle 
import numpy as np 

f = open("midi_data.bin", "rb")
midi_dict = pickle.load(f)
f.close() 

notes = midi_dict['notes'] 
max_ps = -np.inf 
min_ps = np.inf 
acai_input = np.zeros((len(notes), 16))
for i, note in enumerate(notes):
    acai_input[i, :] = note 
    max_ps = max(max(note), max_ps)
    min_ps = min(min(note), min_ps)

print("Max pitch: {}".format(max_ps))
print("Min pitch: {}".format(min_ps))

f = open("midi_input.bin", "wb") 
pickle.dump(acai_input, f)
f.close() 