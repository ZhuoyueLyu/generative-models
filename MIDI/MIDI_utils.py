"""
Part of this code draws from the script available in https://zhuanlan.zhihu.com/p/109194646
"""
from music21.note import Note 
from music21.chord import Chord 
from music21.stream import Part 
import pickle 

def parse_score(midi_score):
    """
    Transform a score object representing the midi file to a list of integers 
    https://web.mit.edu/music21/doc/moduleReference/moduleStream.html#music21.stream.Score
    """
    has_piano = False 
    for part in midi_score.parts:
        if "Piano" == part.partName:
            has_piano = True
            events = part.recurse() # Getting all of the events (note or chord or otherwise) 
    if not has_piano:
        return [] 
    
    parsed_notes = [] # list of integers representing note 
    for event in events:
        if isinstance(event, Note):
            parsed_notes.append(event.pitches[0].ps)
        elif isinstance(event, Chord):
            for note in event.notes:
                parsed_notes.append(note.pitches[0].ps)

    return parsed_notes

def get_midi_data():
    f = open("midi_input.bin", "rb") 
    data = pickle.load(f)
    f.close() 
    return data 
    
if __name__ == "__main__":
    from music21 import instrument, converter
    fname = "clean_midi/3T/Why.mid"
    sc = instrument.partitionByInstrument(converter.parse(fname))
    notes = parse_score(sc)
    print(notes)
    print(len(notes))
