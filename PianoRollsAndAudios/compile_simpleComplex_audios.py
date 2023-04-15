import sys
sys.path.insert(0, '../')

from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, \
    DUALIZATION_ROLAND_HAND_DRUM_MIXED
import os, glob
from bokeh.models import Tabs, Panel
from bokeh.io import show, save, output_file
from bokeh.layouts import layout, column, row


def save_repetitions_to_audio(root_path, save_directory, num_participants, num_repetitions):
    examples = glob.glob(os.path.join(root_path, "*"))
    final_tabs = []
    for ix, test_sample in enumerate(examples):
        print("test_sample :", test_sample)
        save_dir = os.path.join(save_directory, "/".join(test_sample.split('/')[2:])).replace("drummer", f"{ix:03d}_drummer")
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        repetition_groups = [[f"Participant_{i}_{j}" for j in ['simple', 'complex']] for i in range(1, num_participants+1)]

        collected_tabs = []
        for repetition_group in repetition_groups:
            repetition_tabs = []
            for rep_ix in range(num_repetitions):
                hvo_seq_temp = midi_to_hvo_sequence(
                    filename=os.path.join(test_sample, repetition_group[rep_ix] + '.mid'),
                    drum_mapping=DUALIZATION_ROLAND_HAND_DRUM,
                    beat_division_factors=[4])
                hvo_seq_temp.adjust_length(32)
                hvo_seq_temp.save_audio(filename=os.path.join(save_dir,repetition_group[rep_ix]+'.wav'),
                                        sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")
                # print(os.path.join(save_directory,repetition_group[rep_ix]+'.wav'))



        original = midi_to_hvo_sequence(
            filename=os.path.join(os.path.join(test_sample, f'original.mid')),
            drum_mapping=ROLAND_REDUCED_MAPPING,
            beat_division_factors=[4])
        original.adjust_length(32)
        original.save_audio(filename=os.path.join(save_dir, 'original.wav'),
                                sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")


if __name__ == "__main__":
    # organize data as dictionaries
    root_path = "./midi_files/SimpleComplex/tested_with_two_participants/"
    save_repetitions_to_audio(
        root_path,
        save_directory="PianoRollsAndAudios/",
        num_participants=2,
        num_repetitions=2)

    root_path = "./midi_files/SimpleComplex/tested_with_Participant_1_Only/"
    save_repetitions_to_audio(
        root_path,
        save_directory="PianoRollsAndAudios/",
        num_participants=1, num_repetitions=2)


    # Finally use sox to downsample the audio files to 16kHz
    # find . -name '*.wav' -execdir sox {} -r 16000 {}_downsampled.wav \; -execdir mv {}_downsampled.wav {} \;

