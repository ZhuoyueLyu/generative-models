from music21 import *

import numpy as np
import os

# 随机数生成
import random

# keras构建深度学习模型
from keras.layers import *
from keras.models import *
import keras.backend as K


def read_midi(file):
    notes = []
    notes_to_parse = None

    # 解析MIDI文件
    midi = converter.parse(file)
    # 基于乐器分组
    s2 = instrument.partitionByInstrument(midi)

    # 遍历所有的乐器
    for part in s2.parts:
        # 只选择钢琴
        if 'Piano' in str(part):
            notes_to_parse = part.recurse()
            # 查找特定元素是音符还是和弦
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


if __name__ == '__main__':
    # 读取所有文件名

    # directory = os.fsencode('./schubert')
    files = [i for i in os.listdir() if os.fsdecode(i).endswith(".mid")]

    print(files)

    # 读取每个midi文件
    all_notes = []
    for i in files:
        all_notes.append(read_midi(i))

    # 所有midi文件的音符和和弦
    notes = [element for notes in all_notes for element in notes]

    # 输入序列的长度
    no_of_timesteps = 128

    n_vocab = len(set(notes))
    pitch = sorted(set(item for item in notes))

    # 为每个note分配唯一的值
    note_to_int = dict((note, number) for number, note in enumerate(pitch))
    # 准备输入和输出序列
    X = []
    y = []
    for notes in all_notes:
        for i in range(0, len(notes) - no_of_timesteps, 1):
            input_ = notes[i:i + no_of_timesteps]
            output = notes[i + no_of_timesteps]
            X.append([note_to_int[note] for note in input_])
            y.append(note_to_int[output])

    X = np.reshape(X, (len(X), no_of_timesteps, 1))
    # 标准化输入
    X = X / float(n_vocab)

    print(X)
