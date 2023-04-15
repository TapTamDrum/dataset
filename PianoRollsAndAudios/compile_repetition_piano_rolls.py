import sys
sys.path.insert(0, '../')

from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, \
    DUALIZATION_ROLAND_HAND_DRUM_MIXED
import os, glob
from bokeh.models import Tabs, Panel
from bokeh.io import show, save, output_file
from bokeh.layouts import layout, column, row


def get_repetition_plots(root_path, filename, num_participants, num_repetitions):
    examples = glob.glob(os.path.join(root_path, "*"))
    final_tabs = []
    test_sample = examples[0]
    for ix, test_sample in enumerate(examples):
        temp = test_sample.split('/')[-1]
        temp_segs = temp.split("_")
        gmd_id = f"{temp_segs[0]}_{temp_segs[-3]}_{temp_segs[-2]}_{temp_segs[-1]}"
        style = temp_segs[1].split("-")[0]

        temp = test_sample.split('/')[-1]
        temp_segs = temp.split("_")

        repetition_groups = [[f"Participant_{i}_repetition_{j}" for j in range(num_repetitions)] for i in range(1, num_participants+1)]

        collected_tabs = []
        for repetition_group in repetition_groups:
            repetition_tabs = []
            for rep_ix in range(num_repetitions):
                hvo_seq_temp = midi_to_hvo_sequence(
                    filename=os.path.join(test_sample, repetition_group[rep_ix] + '.mid'),
                    drum_mapping=DUALIZATION_ROLAND_HAND_DRUM,
                    beat_division_factors=[4])
                hvo_seq_temp.metadata.update({
                    "gmd_id": gmd_id,
                    "style": style,
                })
                hvo_seq_temp.adjust_length(32)
                repetition_tabs.append(Panel(child=hvo_seq_temp.to_html_plot(filename=repetition_group[rep_ix]),
                                             title=f"Repetition {rep_ix}"))

            collected_tabs.append(Panel(child=Tabs(tabs=repetition_tabs),
                                        title=f"Participant {repetition_group[0].split('_')[1]}"))

        repetitionPanel = Tabs(tabs=collected_tabs)

        original = midi_to_hvo_sequence(
            filename=os.path.join(os.path.join(test_sample, f'original.mid')),
            drum_mapping=ROLAND_REDUCED_MAPPING,
            beat_division_factors=[4])
        original.adjust_length(32)
        original.metadata.update({
            "gmd_id": gmd_id,
            "style": style,
        })
        originalPanel = original.to_html_plot(filename=f"original: {test_sample.split('/')[-1]}.mid")

        final_tabs.append(Panel(child=column(originalPanel, repetitionPanel),
                                title=f"{ix}"))
    # tile original above and then place the repetitions below\
    final_tabs = Tabs(tabs=final_tabs)
    output_file(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save(final_tabs, filename=filename)

    return final_tabs

if __name__ == "__main__":
    # organize data as dictionaries
    root_path = "./midi_files/Repetitions/tested_with_four_participants/"
    final_tabs = get_repetition_plots(root_path, filename="PianoRollsAndAudios/repetition_plots/tested_with_four_participants.html", num_participants=4, num_repetitions=3)

    root_path = "./midi_files/Repetitions/tested_with_Participant_1_Only/"
    final_tabs = get_repetition_plots(root_path, filename="PianoRollsAndAudios/repetition_plots/tested_with_Participant_1_Only.html", num_participants=1, num_repetitions=3)



