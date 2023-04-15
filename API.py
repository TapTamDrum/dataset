from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, \
    ROLAND_REDUCED_MAPPING_HEATMAPS
import os
import glob
import copy
from eval.GrooveEvaluator import Evaluator
from bokeh.models import Tabs
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Panel, Range1d, HoverTool

import typing
from random import sample
try:
    import pretty_midi

    _HAS_PRETTY_MIDI = True
except ImportError:
    _HAS_PRETTY_MIDI = False

try:
    from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
    from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, \
        ROLAND_REDUCED_MAPPING_HEATMAPS

    _HAS_HVO_SEQUENCE = True
except ImportError:
    _HAS_HVO_SEQUENCE = False

try:
    import note_seq

    _HAS_NOTE_SEQ = True
except ImportError:
    _HAS_NOTE_SEQ = False

try:
    from IPython.display import Audio
    from IPython.core.display import display
    _HAS_AUDIO_PLAYER = True
except ImportError:
    _HAS_AUDIO_PLAYER = False

from bokeh.models import Tabs, Panel
from bokeh.io import show
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

class DualizationDatasetAPI:
    def __init__(self, midi_folder=None):
        self.__summary_dataframe = pd.DataFrame()

        # self.__midi_folder = ""
        # test_folders = []
        # self.__test_numbers = []
        # self.__wasTestedOnP1 = []
        # self.__wasTestedOnP2 = []
        # self.__wasTestedOnP3 = []
        # self.__wasTestedOnP4 = []
        # self.__wasTestedOnMultipleParticipants = []
        # self.__testType = []
        # self.__testNumber = []
        # self.__style = []
        # self.__tempo = []
        # self.__gmdDrummer = []
        # self.__gmdPerformanceSession = []
        # self.__gmdSegmentType = []
        # self.__gmdSegmentMeter = []
        # self.__selected2BarsFromStart = []
        # self.__dualizedMidifolderPath = []

        if midi_folder is not None:
            self.populate_fields(midi_folder)

    def populate_fields(self, midi_folder):
        tests = glob.glob(f'{midi_folder}/**/original.mid', recursive=True)
        assert len(tests) > 0, "No folders found in {}".format(midi_folder)
        test_folders = sorted([os.path.dirname(f) for f in tests], key=lambda x: x.split(os.sep)[-1])
        print("Found {} tested patterns".format(len(test_folders)))

        self.__summary_dataframe["Test Number"] = [t.split(os.sep)[-1].split(" ")[0].split("[")[-1] for t in
                                                   test_folders]
        self.__summary_dataframe["Was Tested On P1"] = [True if "P1" in t else False for t in test_folders]
        self.__summary_dataframe["Was Tested On P2"] = [True if "P2" in t else False for t in test_folders]
        self.__summary_dataframe["Was Tested On P3"] = [True if "P3" in t else False for t in test_folders]
        self.__summary_dataframe["Was Tested On P4"] = [True if "P4" in t else False for t in test_folders]
        self.__summary_dataframe["Was Tested On Multiple Participants"] = [
            True if len(t.split(" ")[1].split("]")[0]) > 2 else False for t in test_folders]
        self.__summary_dataframe["Test Type"] = [
            t.split(os.sep)[1].replace("Repetitions", "Three Random Repetitions").replace(
                "SimpleComplex", "Simple Complex") for t in test_folders]
        self.__summary_dataframe["Test Number"] = [t.split(os.sep)[-1].split(" ")[0].split("[")[-1] for t in
                                                   test_folders]
        self.__summary_dataframe["Style"] = [t.split(os.sep)[-1].split(" ")[-1].split("_")[1].lower() for t in
                                             test_folders]
        self.__summary_dataframe["Tempo"] = [float(t.split(os.sep)[-1].split(" ")[-1].split("_")[2]) for t in
                                             test_folders]
        self.__summary_dataframe["GMD Drummer"] = [t.split(os.sep)[-1].split(" ")[-1].split("-")[0] for t in
                                                   test_folders]
        self.__summary_dataframe["GMD Performance Session"] = [
            t.split(os.sep)[-1].split(" ")[-1].split("-")[1].split("_")[0]
            for t in test_folders]
        self.__summary_dataframe["GMD Segment Type"] = [t.split(os.sep)[-1].split(" ")[-1].split("_")[3] for t in
                                                        test_folders]
        self.__summary_dataframe["GMD Segment Meter"] = [t.split(os.sep)[-1].split(" ")[-1].split("_")[4] for t in
                                                         test_folders]
        self.__summary_dataframe["Selected 2Bars From Start"] = [int(t.split(os.sep)[-1].split(" ")[-1].split("_")[-1])
                                                                 for t in
                                                                 test_folders]
        self.__summary_dataframe["Dualized Midifolder Path"] = [os.path.join(os.path.dirname(t), "dualized") for t in
                                                                test_folders]
        self.__summary_dataframe["Test Folder"] = test_folders

        # remove indices
        self.__summary_dataframe.reset_index()

    def __repr__(self):
        str_ = "------" * 10
        str_ += f"\nNumber of Drum Patterns Dualized --> {len(self.__summary_dataframe.index)}"
        str_ += f"\nfields available: {self.__summary_dataframe.columns.__str__()}\n"
        str_ += "------" * 10
        return str_

    def __len__(self):
        return len(self.__summary_dataframe.index)

    def __getitem__(self, items):

        if not isinstance(items, list):
            if not isinstance(items, tuple):
                items = [items]
            else:
                items = [item for item in items]

        dualizationForTestsList = []

        for item in items:
            item = self.__format_test_number(item)
            dualizationForTestsList.append(DualizationTest(self, item))

        return dualizationForTestsList if len(dualizationForTestsList) > 1 else dualizationForTestsList[0]

    def __iter__(self):
        for test_number in self.__summary_dataframe["Test Number"].values:
            yield DualizationTest(self, test_number)

    @property
    def dualization_tests(self):
        return [DualizationTest(self, test_number) for test_number in self.__summary_dataframe["Test Number"].values]

    def copy(self):
        return copy.deepcopy(self)

    def __remove_datapoints(self, should_keep_):
        """ Removes datapoints from the dataset
        @param should_keep_:  [True, False, True, ...] indicating which datapoints to keep
        @return: None
        """
        new_dataframe = pd.DataFrame()
        # remove rows from self.__summary_dataframe if should_keep_ is false
        for col in self.__summary_dataframe.columns:
            new_dataframe[col] = self.__summary_dataframe[col][should_keep_]
        self.__summary_dataframe = new_dataframe
        self.__summary_dataframe.reset_index(drop=True, inplace=True)


    @property
    def fields(self):
        return self.__summary_dataframe.columns

    @property
    def summary_dataframe(self):
        return self.__summary_dataframe

    @property
    def summary_dict(self):
        return self.__summary_dataframe.to_dict(orient="records")

    def save_summary(self, filename):
        summary = self.summary_dataframe
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        summary.to_csv(filename, index=False)
        print("Saved csv to {}".format(filename))

    @property
    def MultipleParticipantSubset(self):
        """ Returns a subset of dataset for which the drum patterns were tested on at least two participants
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On Multiple Participants"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def SingleParticipantSubset(self):
        """ Returns a subset of dataset for which the drum patterns were tested on a single participant only
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On Multiple Participants"] == False
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def ThreeRepetitionSubset(self):
        """ Returns a subset object with only the tests in which a given drum pattern was presented
        three times randomly to the participants without letting them know that they had already
        dualized the pattern before
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Test Type"] == "Three Random Repetitions"
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def SimpleComplexSubset(self):
        """ Returns a subset object with only the tests in which the participants were
        asked to provide a simple and a complex dualization for a given drum pattern
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Test Type"] == "Simple Complex"
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P1Subset(self):
        """ Returns a subset object with only the tests that were performed on P1
        @return: DualizationDatasetAPI object """
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P1"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P2Subset(self):
        """ Returns a subset object with only the tests that were performed on P2
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P2"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P3Subset(self):
        """ Returns a subset object with only the tests that were performed on P3
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P3"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P4Subset(self):
        """ Returns a subset object with only the tests that were performed on P4
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P4"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    def get_subset_matching_styles(self, style, hard_match=False):
        """
        Filters the dataset by style (or styles) and returns a new dataset.
        The matching ignores case.
        @param style:
        @param hard_match:
        @return:
        """
        new_dataset = self.copy()
        if isinstance(style, str):
            style = [style]

        style = [s.lower() for s in style]

        if hard_match:
            indices = self.summary_dataframe["Style"].isin(style)
        else:
            indices = []
            for s in style:
                indices.append(self.summary_dataframe["Style"].str.lower().str.contains(s))
            indices = np.any(indices, axis=0)
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    def get_subset_excluding_styles(self, style, hard_match=False):
        """
        Filters the dataset by style (or styles) and returns a new dataset.
        The matching ignores case.
        @param style:
        @param hard_match:
        @return:
        """
        new_dataset = self.copy()
        if isinstance(style, str):
            style = [style]

        style = [s.lower() for s in style]

        if hard_match:
            indices = self.summary_dataframe["Style"].isin(style)
        else:
            indices = []
            for s in style:
                indices.append(self.summary_dataframe["Style"].str.lower().str.contains(s))
            indices = np.any(indices, axis=0)
        new_dataset.__remove_datapoints(~indices)
        return new_dataset

    def get_subset_within_tempo_range(self, min_, max_):
        """
        Filters the dataset by tempo range and returns a new dataset.
        @param min_:  minimum tempo
        @param max_:  maximum tempo
        @return:  new dataset
        """
        new_dataset = self.copy()
        indices = (self.summary_dataframe["Tempo"] >= min_) & (self.summary_dataframe["Tempo"] <= max_)
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    def __format_test_number(self, test_number, check_if_exists=True):
        """
        Formats the test number to 3 digits (e.g. 1 -> 001)
        @return:
        """
        df = self.summary_dataframe
        test_number = f"{test_number:03d}" if isinstance(test_number, int) else test_number
        if test_number not in df["Test Number"].values:
            raise ValueError(f"Test number {test_number} not found in dataset. Select from {df['Test Number'].values}")
        return test_number

    def get_participants_attempted_test_number(self, test_number):
        """
        Returns a list of participants who attempted a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  list of participants
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        participants = []
        if row["Was Tested On P1"].values[0]:
            participants.append(1)
        if row["Was Tested On P2"].values[0]:
            participants.append(2)
        if row["Was Tested On P3"].values[0]:
            participants.append(3)
        if row["Was Tested On P4"].values[0]:
            participants.append(4)
        return participants

    def get_test_numbers(self):
        """
        Returns a list of test numbers
        @return:  list of test numbers
        """
        return self.summary_dataframe["Test Number"].values

    def get_folder_path_for_test_number(self, test_number):
        """
        Returns the folder path for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  path to folder containing the test number
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Test Folder"].values[0]

    def get_tested_drum_pattern_path(self, test_number):
        """
        Returns the midi file for the drum pattern that was tested
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  midi file
        """
        return os.path.join(self.get_folder_path_for_test_number(test_number), "original.mid")

    def get_test_type_for_test_number(self, test_number):
        """
        Returns the test type for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  test type
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Test Type"].values[0]

    def get_style_for_test_number(self, test_number):
        """
        Returns the style for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  style
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Style"].values[0]

    def get_tempo_for_test_number(self, test_number):
        """
        Returns the tempo for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  tempo
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Tempo"].values[0]

    def get_participant_dualizations_for_test_number(self, test_number, participant):
        """
        Returns a DualizationsForParticipant object for a given test number and participant
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  DualizationsForParticipant object
        """
        test_number = self.__format_test_number(test_number)
        if participant not in self.get_participants_attempted_test_number(test_number):
            raise ValueError(f"Participant {participant} did not attempt test number {test_number}")
        dualizationData = ParticipantDualizations()
        dualizationData.populate_attributes(self, test_number, participant)
        return dualizationData

    def piano_rolls(self, show_=False, ):
        _4P_ThreeReps = self.ThreeRepetitionSubset.MultipleParticipantSubset

        _P1_ThreeReps = self.ThreeRepetitionSubset.SingleParticipantSubset

        _2P_SimpleComplex = self.SimpleComplexSubset.MultipleParticipantSubset

        _P1_SimpleComplex = self.SimpleComplexSubset.SingleParticipantSubset

        tabs_ = []
        if len(_4P_ThreeReps) > 0:
            t_ = []
            for dualizationtest in _4P_ThreeReps.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.piano_rolls(show_=False), title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="3 Repetitions with P1 to P4"))

        if len(_P1_ThreeReps) > 0:
            t_ = []
            for dualizationtest in _P1_ThreeReps.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.piano_rolls(show_=False), title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="3 Repetitions with P1"))

        if len(_2P_SimpleComplex) > 0:
            t_ = []
            for dualizationtest in _2P_SimpleComplex.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.piano_rolls(show_=False), title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="Simple/Complex with P1 to P2"))

        if len(_P1_SimpleComplex) > 0:
            t_ = []
            for dualizationtest in _P1_SimpleComplex.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.piano_rolls(show_=False), title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="Simple/Complex with P1"))

        print("Compiling tabs...")
        tabs = Tabs(tabs=tabs_)
        print("Compiled tabs.")

        if show_:
            show(tabs)

        return tabs

    def z_plots(self, show_=False, quantize_all=False, flatten_all=False,
                color_with_vel_values=False, show_balance_evenness_entropy=False, scale_plot=1.0):
        _4P_ThreeReps = self.ThreeRepetitionSubset.MultipleParticipantSubset

        _P1_ThreeReps = self.ThreeRepetitionSubset.SingleParticipantSubset

        _2P_SimpleComplex = self.SimpleComplexSubset.MultipleParticipantSubset

        _P1_SimpleComplex = self.SimpleComplexSubset.SingleParticipantSubset

        tabs_ = []
        if len(_4P_ThreeReps) > 0:
            t_ = []
            for dualizationtest in _4P_ThreeReps.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.z_plots(show_=False, quantize_all=quantize_all,
                                                             flatten_all=flatten_all,
                                                             color_with_vel_values=color_with_vel_values,
                                                             show_balance_evenness_entropy=show_balance_evenness_entropy,
                                                             scale_plot=scale_plot),
                                title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="3 Repetitions with P1 to P4"))

        if len(_P1_ThreeReps) > 0:
            t_ = []
            for dualizationtest in _P1_ThreeReps.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.z_plots(show_=False, quantize_all=quantize_all,
                                                             flatten_all=flatten_all,
                                                             color_with_vel_values=color_with_vel_values,
                                                             show_balance_evenness_entropy=show_balance_evenness_entropy,
                                                             scale_plot=scale_plot),
                                title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="3 Repetitions with P1"))

        if len(_2P_SimpleComplex) > 0:
            t_ = []
            for dualizationtest in _2P_SimpleComplex.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.z_plots(show_=False, quantize_all=quantize_all,
                                                             flatten_all=flatten_all,
                                                             color_with_vel_values=color_with_vel_values,
                                                             show_balance_evenness_entropy=show_balance_evenness_entropy,
                                                             scale_plot=scale_plot),
                                title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="Simple/Complex with P1 to P2"))

        if len(_P1_SimpleComplex) > 0:
            t_ = []
            for dualizationtest in _P1_SimpleComplex.dualization_tests:
                print(f"Processing {dualizationtest.short_name}...")
                t_.append(Panel(child=dualizationtest.z_plots(show_=False, quantize_all=quantize_all,
                                                             flatten_all=flatten_all,
                                                             color_with_vel_values=color_with_vel_values,
                                                             show_balance_evenness_entropy=show_balance_evenness_entropy,
                                                             scale_plot=scale_plot),
                                title=dualizationtest.short_name))
            tabs_.append(Panel(child=Tabs(tabs=t_), title="Simple/Complex with P1"))

        print("Compiling tabs...")
        tabs = Tabs(tabs=tabs_)
        print("Compiled tabs.")

        if show_:
            show(tabs)

        return tabs

    def calculate_intra_dualization_edit_distances(self, normalize_by_union=False):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.calculate_intra_dualization_edit_distances(normalize_by_union=normalize_by_union)
            for key, value in res_dict.items():
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].extend(value)
        return results_dict

    def calculate_inter_dualization_edit_distances(self, normalize_by_union=False):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.calculate_inter_dualization_edit_distances(normalize_by_union=normalize_by_union)
            if res_dict is not None:
                for key, value in res_dict.items():
                    if key not in results_dict:
                        results_dict[key] = []
                    results_dict[key].extend(value)
        return results_dict

    def calculate_intra_dualization_jaccard_similarities(self):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.calculate_intra_dualization_jaccard_similarities()
            for key, value in res_dict.items():
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].extend(value)
        return results_dict

    def calculate_intra_dualization_cohens_kappas(self):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.calculate_intra_dualization_cohens_kappas()
            for key, value in res_dict.items():
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].extend(value)
        return results_dict

    def calculate_inter_dualization_cohens_kappas(self):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.calculate_inter_dualization_cohens_kappas()
            if res_dict is not None:
                for key, value in res_dict.items():
                    if key not in results_dict:
                        results_dict[key] = []
                    results_dict[key].extend(value)
        return results_dict


    def calculate_inter_dualization_jaccard_similarities(self):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.calculate_inter_dualization_jaccard_similarities()
            if res_dict is not None:
                for key, value in res_dict.items():
                    if key not in results_dict:
                        results_dict[key] = []
                    results_dict[key].extend(value)
        return results_dict

    def collect_step_densities(self, normalize_by_original=False):
        results_dict = {}
        for dualizationtest in self.dualization_tests:
            res_dict = dualizationtest.collect_step_densities(normalize_by_original=normalize_by_original)
            for key, value in res_dict.items():
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].extend(value)
        return results_dict

    def get_n_random_dualization_pairs_from_three_repetitions(self, n_sample, randomize_scores=False):
        three_repetitions = self.ThreeRepetitionSubset.MultipleParticipantSubset
        if len(three_repetitions) == 0:
            return None
        else:
            pairs = []
            dualization_tests = self.dualization_tests
            for n_sample in range(n_sample):
                dualization_test1 = sample(dualization_tests, 1)[0]
                dualization_test2 = sample(dualization_tests, 1)[0]
                # select 1 on four participants
                participant_ids = [1, 2, 3, 4]
                participant_id1 = sample(participant_ids, 1)[0]
                rep_id1 = sample([0, 1, 2], 1)[0]
                participant_ids.remove(participant_id1)
                participant_id2 = sample(participant_ids, 1)[0]
                rep_id2 = sample([0, 1, 2], 1)[0]
                patter1 = eval(f"dualization_test1.P{participant_id1}.rep{rep_id1}")
                patter2 = eval(f"dualization_test2.P{participant_id2}.rep{rep_id2}")
                if randomize_scores:
                    patter1.randomize_hits()
                    patter2.randomize_hits()
                pairs.append((patter1, patter2))
            return pairs

    def get_n_random_simple_pairs_from_simple_complex(self, n_sample, randomize_scores=False):
        simplecomplex_repetitions = self.SimpleComplexSubset.MultipleParticipantSubset
        if len(simplecomplex_repetitions) == 0:
            return None
        else:
            pairs = []
            dualization_tests = self.dualization_tests
            for n_sample in range(n_sample):
                dualization_test1 = sample(dualization_tests, 1)[0]
                dualization_test2 = sample(dualization_tests, 1)[0]
                # select 1 on four participants
                participant_ids = [1, 2]
                participant_id1 = sample(participant_ids, 1)[0]
                rep_id1 = sample([0, 1, 2], 1)[0]
                participant_ids.remove(participant_id1)
                participant_id2 = sample(participant_ids, 1)[0]
                rep_id2 = sample([0, 1, 2], 1)[0]
                patter1 = dualization_test1.P1.simple
                patter2 = dualization_test2.P2.simple
                if randomize_scores:
                    patter1.randomize_hits()
                    patter2.randomize_hits()
                pairs.append((patter1, patter2))
            return pairs

    def get_n_random_complex_pairs_from_simple_complex(self, n_sample, randomize_scores=False):
        simplecomplex_repetitions = self.SimpleComplexSubset.MultipleParticipantSubset
        if len(simplecomplex_repetitions) == 0:
            return None
        else:
            pairs = []
            dualization_tests = self.dualization_tests
            for n_sample in range(n_sample):
                dualization_test1 = sample(dualization_tests, 1)[0]
                dualization_test2 = sample(dualization_tests, 1)[0]
                # select 1 on four participants
                participant_ids = [1, 2]
                participant_id1 = sample(participant_ids, 1)[0]
                rep_id1 = sample([0, 1, 2], 1)[0]
                participant_ids.remove(participant_id1)
                participant_id2 = sample(participant_ids, 1)[0]
                rep_id2 = sample([0, 1, 2], 1)[0]
                patter1 = dualization_test1.P1.complex
                patter2 = dualization_test2.P2.complex
                if randomize_scores:
                    patter1.randomize_hits()
                    patter2.randomize_hits()
                pairs.append((patter1, patter2))
            return pairs

    def get_n_random_simplecomplex_pairs_from_simple_complex(self, n_sample, randomize_scores=False):
        simplecomplex_repetitions = self.SimpleComplexSubset.MultipleParticipantSubset
        if len(simplecomplex_repetitions) == 0:
            return None
        else:
            pairs = []
            dualization_tests = self.dualization_tests
            for n_sample in range(n_sample):
                dualization_test1 = sample(dualization_tests, 1)[0]
                dualization_test2 = sample(dualization_tests, 1)[0]
                # select 1 on four participants
                participant_ids = [1, 2]
                participant_id1 = sample(participant_ids, 1)[0]
                vals = ["simple", "complex"]
                rep_id1 = sample([0, 1], 1)[0]
                participant_ids.remove(participant_id1)
                patter1 = eval(f"dualization_test1.P1.{vals[rep_id1]}")
                patter2 = eval(f"dualization_test2.P2.{vals[1-rep_id1]}")
                if randomize_scores:
                    patter1.randomize_hits()
                    patter2.randomize_hits()
                pairs.append((patter1, patter2))
            return pairs

    def get_n_random_simplecomplex_pairs_from_simple_complex_single_participant(self, n_sample, participant,
                                                                                randomize_scores=False):
        simplecomplex_repetitions = self.SimpleComplexSubset.MultipleParticipantSubset
        if len(simplecomplex_repetitions) == 0:
            return None
        else:
            pairs = []
            dualization_tests = self.dualization_tests
            for n_sample in range(n_sample):
                dualization_test1 = sample(dualization_tests, 1)[0]
                dualization_test2 = sample(dualization_tests, 1)[0]
                # select 1 on four participants
                vals = ["simple", "complex"]
                rep_id1 = sample([0, 1], 1)[0]
                patter1 = eval(f"dualization_test1.P{participant}.{vals[rep_id1]}")
                patter2 = eval(f"dualization_test2.P{participant}.{vals[1-rep_id1]}")
                if randomize_scores:
                    patter1.randomize_hits()
                    patter2.randomize_hits()
                pairs.append((patter1, patter2))
            return pairs

    def extract_inter_edit_distances_from_list_of_pattern_pairs(self, pattern_pairs, normalize_by_union=False):
        results = []
        for pattern_pair in pattern_pairs:
            pattern1 = pattern_pair[0]
            pattern2 = pattern_pair[1]
            results.append(pattern1.calculate_edit_distance_with(pattern2, normalize_by_union=normalize_by_union))
        return results

    def extract_inter_jaccard_similarities_from_list_of_pattern_pairs(self, pattern_pairs):
        results = []
        for pattern_pair in pattern_pairs:
            pattern1 = pattern_pair[0]
            pattern2 = pattern_pair[1]
            results.append(pattern1.calculate_jaccard_similarity_with(pattern2))
        return results

    def extract_inter_cohen_kappa_from_list_of_pattern_pairs(self, pattern_pairs):
        results = []
        for pattern_pair in pattern_pairs:
            pattern1 = pattern_pair[0]
            pattern2 = pattern_pair[1]
            results.append(pattern1.calculate_cohens_kappa_with(pattern2))
        return results


class DualizationTest:
    def __init__(self, dualizationDataset=None, test_number=None):
        self.__Tempo = None
        self.__Style = None
        self.__TestType = None
        self.__TestNumber = None
        self.__FolderPath = None
        self.__P1 = None
        self.__P2 = None
        self.__P3 = None
        self.__P4 = None
        if dualizationDataset is not None and test_number is not None:
            self.populate_attributes(dualizationDataset, test_number)

    def populate_attributes(self, dualizationDataset, test_number):
        self.__FolderPath = dualizationDataset.get_folder_path_for_test_number(test_number)
        self.__TestNumber = test_number
        self.__TestType = dualizationDataset.get_test_type_for_test_number(test_number)
        self.__Style = dualizationDataset.get_style_for_test_number(test_number)
        self.__Tempo = dualizationDataset.get_tempo_for_test_number(test_number)

        if 1 in dualizationDataset.get_participants_attempted_test_number(test_number):
            self.__P1 = dualizationDataset.get_participant_dualizations_for_test_number(test_number, 1)
        if 2 in dualizationDataset.get_participants_attempted_test_number(test_number):
            self.__P2 = dualizationDataset.get_participant_dualizations_for_test_number(test_number, 2)
        if 3 in dualizationDataset.get_participants_attempted_test_number(test_number):
            self.__P3 = dualizationDataset.get_participant_dualizations_for_test_number(test_number, 3)
        if 4 in dualizationDataset.get_participants_attempted_test_number(test_number):
            self.__P4 = dualizationDataset.get_participant_dualizations_for_test_number(test_number, 4)

    @property
    def short_name(self):
        return " ".join(self.FolderPath.split(os.sep)[-1].split(" ")[:2])

    @property
    def FolderPath(self):
        return self.__FolderPath

    @property
    def TestNumber(self):
        return self.__TestNumber

    @property
    def TestType(self):
        return self.__TestType

    @property
    def Style(self):
        return self.__Style

    @property
    def Tempo(self):
        return self.__Tempo

    @property
    def P1(self):
        if self.__P1 is None:
            raise ValueError("Participant 1 did not attempt this test")
        return self.__P1

    @property
    def P2(self):
        if self.__P2 is None:
            raise ValueError("Participant 2 did not attempt this test")
        return self.__P2

    @property
    def P3(self):
        if self.__P3 is None:
            raise ValueError("Participant 3 did not attempt this test")
        return self.__P3

    @property
    def P4(self):
        if self.__P4 is None:
            raise ValueError("Participant 4 did not attempt this test")
        return self.__P4

    @property
    def isMultiParticipant(self):
        # check if more than one participant attempted this test
        participants_attempted_test = self.compile_dualizations_of_participants_attempted_test()
        if len(participants_attempted_test) > 1:
            return True

    def compile_dualizations_of_participants_attempted_test(self):
        participants = []
        if self.__P1 is not None:
            participants.append(self.__P1)
        if self.__P2 is not None:
            participants.append(self.__P2)
        if self.__P3 is not None:
            participants.append(self.__P3)
        if self.__P4 is not None:
            participants.append(self.__P4)
        return participants

    def piano_rolls(self, show_=False):
        tabs = []
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            tabs.append(
                Panel(child=participantDualization.piano_rolls(),
                      title=f"Participant {participantDualization.Participant}")
            )
        tabs = Tabs(tabs=tabs)
        if show_:
            show(tabs)
        return tabs

    def z_plots(self, show_=False, quantize_all=False, flatten_all=False,
                color_with_vel_values=False, show_balance_evenness_entropy=False, scale_plot=1.0):
        tabs = []
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            tabs.append(
                Panel(child=participantDualization.z_plots(
                    quantize_all=quantize_all, flatten_all=flatten_all,
                    color_with_vel_values=color_with_vel_values,
                    show_balance_evenness_entropy=show_balance_evenness_entropy,
                    scale_plot=scale_plot
                ),
                    title=f"Participant {participantDualization.Participant}")
            )
        tabs = Tabs(tabs=tabs)
        if show_:
            show(tabs)
        return tabs

    def audios(self):
        audios_ = {}
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            audios_[f"Participant {participantDualization.Participant}"] = participantDualization.audios()
        return audios_

    def calculate_intra_dualization_edit_distances(self, normalize_by_union=False):
        results_dict = {}
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            res = participantDualization.calculate_intra_dualization_edit_distances(
                normalize_by_union=normalize_by_union)
            if participantDualization.TestType == "Simple Complex":
                results_dict[f"Participant {participantDualization.Participant} Simple vs. " \
                             f"Participant {participantDualization.Participant} Complex"] = res
            elif participantDualization.TestType == "Three Random Repetitions":
                results_dict[f"Participant {participantDualization.Participant}"] = res
        return results_dict

    def calculate_inter_dualization_edit_distances(self, normalize_by_union=False):
        if not self.isMultiParticipant:
            return None

        participants_attempted_test = self.compile_dualizations_of_participants_attempted_test()

        results_dict = {}

        for i in range(len(participants_attempted_test)):
            for j in range(i + 1, len(participants_attempted_test)):
                res = participants_attempted_test[i].calculate_inter_dualization_edit_distances(
                    participants_attempted_test[j], normalize_by_union=normalize_by_union)

                if self.TestType == "Three Random Repetitions":
                    results_dict[f"Participant {participants_attempted_test[i].Participant} vs. "
                                 f"Participant {participants_attempted_test[j].Participant}"] = res
                elif self.TestType == "Simple Complex":
                    results_dict[f"Participant {participants_attempted_test[i].Participant} Simple vs. " \
                                 f"Participant {participants_attempted_test[j].Participant} Simple"] = [res["Simple"]]
                    results_dict[f"Participant {participants_attempted_test[i].Participant} Complex vs. "
                                 f"Participant {participants_attempted_test[j].Participant} Complex"] = [res["Complex"]]

        return results_dict

    def calculate_intra_dualization_jaccard_similarities(self):
        results_dict = {}
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            res = participantDualization.calculate_intra_dualization_jaccard_similarities()
            if participantDualization.TestType == "Simple Complex":
                results_dict[f"Participant {participantDualization.Participant} Simple vs. " 
                             f"Participant {participantDualization.Participant} Complex"] = res
            elif participantDualization.TestType == "Three Random Repetitions":
                results_dict[f"Participant {participantDualization.Participant}"] = res
        return results_dict

    def calculate_inter_dualization_jaccard_similarities(self):
        if not self.isMultiParticipant:
            return None

        participants_attempted_test = self.compile_dualizations_of_participants_attempted_test()

        results_dict = {}

        for i in range(len(participants_attempted_test)):
            for j in range(i + 1, len(participants_attempted_test)):
                res = participants_attempted_test[i].calculate_inter_dualization_jaccard_similarities(
                    participants_attempted_test[j])

                if self.TestType == "Three Random Repetitions":
                    results_dict[f"Participant {participants_attempted_test[i].Participant} vs. "
                                 f"Participant {participants_attempted_test[j].Participant}"] = res
                elif self.TestType == "Simple Complex":
                    results_dict[f"Participant {participants_attempted_test[i].Participant} Simple vs. "
                                 f"Participant {participants_attempted_test[j].Participant} Simple"] = [res["Simple"]]
                    results_dict[f"Participant {participants_attempted_test[i].Participant} Complex vs."
                                 f"Participant {participants_attempted_test[j].Participant} Complex"] = [res["Complex"]]

        return results_dict

    def calculate_intra_dualization_cohens_kappas(self):
        results_dict = {}
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            res = participantDualization.calculate_intra_dualization_cohens_kappa()
            if participantDualization.TestType == "Simple Complex":
                results_dict[f"Participant {participantDualization.Participant} Simple vs. " \
                             f"Participant {participantDualization.Participant} Complex"] = res
            elif participantDualization.TestType == "Three Random Repetitions":
                results_dict[f"Participant {participantDualization.Participant}"] = res
        return results_dict

    def calculate_inter_dualization_cohens_kappas(self):
        if not self.isMultiParticipant:
            return None

        participants_attempted_test = self.compile_dualizations_of_participants_attempted_test()

        results_dict = {}

        for i in range(len(participants_attempted_test)):
            for j in range(i + 1, len(participants_attempted_test)):
                res = participants_attempted_test[i].calculate_inter_dualization_cohens_kappa(
                    participants_attempted_test[j])

                if self.TestType == "Three Random Repetitions":
                    results_dict[f"Participant {participants_attempted_test[i].Participant} vs. "
                                 f"Participant {participants_attempted_test[j].Participant}"] = res
                elif self.TestType == "Simple Complex":
                    results_dict[f"Participant {participants_attempted_test[i].Participant} Simple vs. "
                                 f"Participant {participants_attempted_test[j].Participant} Simple"] = [res["Simple"]]
                    results_dict[f"Participant {participants_attempted_test[i].Participant} Complex vs. "
                                 f"Participant {participants_attempted_test[j].Participant} Complex"] = [res["Complex"]]

        return results_dict


    def collect_step_densities(self, normalize_by_original=False):
        results_dict = {}
        participantDualizations = self.compile_dualizations_of_participants_attempted_test()
        for participantDualization in participantDualizations:
            res = participantDualization.collect_step_densities(normalize_by_original=normalize_by_original)
            if participantDualization.TestType == "Simple Complex":
                results_dict[f"Participant {participantDualization.Participant} Simple"] = [res["Simple"]]
                results_dict[f"Participant {participantDualization.Participant} Complex"] = [res["Complex"]]
            elif participantDualization.TestType == "Three Random Repetitions":
                results_dict[f"Participant {participantDualization.Participant}"] = res
        return results_dict

    # def calculate_inter_step_density_distances(self, other_ParticipantDualizations,
    #                                            normalize_by_union=False, throw_error_if_not_same_test_type=True):
    #


class ParticipantDualizations:
    def __int__(self):
        self.__TestNumber = None  # 001, 002, 003, ... (str)
        self.__FolderPath = None  # path to the folder containing the test
        self.__Participant = None  # 1, 2, 3 or 4 (int) for the participants P1, P2, P3 or P4
        self.__TestType = None  # "Three Random Repetitions" or "Simple Complex"
        self.__otherParticipants = None  # list of other participants if multiple participants tried the test pattern
        self.__rep0 = None  # 1st repetition if test type is "Three Random Repetitions"
        self.__rep1 = None  # 2nd repetition if test type is "Three Random Repetitions"
        self.__rep2 = None  # 3rd repetition if test type is "Three Random Repetitions"
        self.__simple = None  # simple if test type is "Simple Complex"
        self.__complex = None  # complex if test type is "Simple Complex"
        self.__style = None  # style of the test pattern
        self.__tempo = None  # tempo of the test pattern
        self.__original = None  # original test pattern

    def __repr__(self):
        str = f"Participant {self.__Participant}, Dualization Test: {self.__TestType}, Style: {self.style}, Tempo: {self.tempo}"
        if self.__TestType == "Three Random Repetitions":
            str += "\n use .rep0, .rep1, .rep2 to access dualizations, \n" \
                   "use .original to access the drum pattern used for dualization"
        else:
            str += "\n use .simple and .complex to access dualizations, \n" \
                   "use .original to access the drum pattern used for dualization"
        return str

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        if self.__TestType == "Three Random Repetitions":
            return iter([self.__rep0, self.__rep1, self.__rep2])
        else:
            return iter([self.__simple, self.__complex])

    def populate_attributes(self, dualizationDatasetAPI, number, participant):
        """
        Populates the attributes of the DualizationsForParticipant class
        @param dualizationDatasetAPI:  DualizationDatasetAPI object
        @param number:  test number
        @param participant:  participant number
        @return:
        """
        self.FolderPath = dualizationDatasetAPI.get_folder_path_for_test_number(number)
        self.TestNumber = number
        self.Participant = participant
        self.TestType = dualizationDatasetAPI.get_test_type_for_test_number(number)
        self.otherParticipants = dualizationDatasetAPI.get_participants_attempted_test_number(number)
        # remove the current participant from the list of other participants
        self.otherParticipants.remove(participant)
        self.style = dualizationDatasetAPI.get_style_for_test_number(number)
        self.tempo = dualizationDatasetAPI.get_tempo_for_test_number(number)
        if self.TestType == "Three Random Repetitions":
            self.rep0 = os.path.join(self.FolderPath, f"Participant_{participant}_repetition_0.mid")
            self.rep1 = os.path.join(self.FolderPath, f"Participant_{participant}_repetition_1.mid")
            self.rep2 = os.path.join(self.FolderPath, f"Participant_{participant}_repetition_2.mid")
        elif self.TestType == "Simple Complex":
            self.simple = os.path.join(self.FolderPath, f"Participant_{participant}_simple.mid")
            self.complex = os.path.join(self.FolderPath, f"Participant_{participant}_complex.mid")
        self.original = os.path.join(self.FolderPath, f"original.mid")

    # get_test_type_for_test_number, get_style_for_test_number, get_tempo_for_test_number
    @property
    def FolderPath(self):
        return self.__FolderPath

    @FolderPath.setter
    def FolderPath(self, value):
        self.__FolderPath = value

    @property
    def short_name(self):
        return " ".join(self.FolderPath.split(os.sep)[-1].split(" ")[:2])

    @property
    def TestNumber(self):
        return self.__TestNumber

    @TestNumber.setter
    def TestNumber(self, value):
        self.__TestNumber = value

    @property
    def Participant(self):
        return self.__Participant

    @Participant.setter
    def Participant(self, value):
        assert value in [1, 2, 3, 4], "Participant must be 1, 2, 3 or 4"
        self.__Participant = value

    @property
    def TestType(self):
        return self.__TestType

    @TestType.setter
    def TestType(self, value):
        assert value in ["Three Random Repetitions", "Simple Complex"], \
            "TestType must be 'Three Random Repetitions' or 'Simple Complex'"
        self.__TestType = value

    @property
    def otherParticipants(self):
        return self.__otherParticipants

    @otherParticipants.setter
    def otherParticipants(self, value):
        if isinstance(value, int):
            value = [value]
        assert isinstance(value, list), "otherParticipants must be a list of integers"
        assert min(value) >= 1 and max(value) <= 4, "otherParticipants must be a list of integers between 1 and 4"
        self.__otherParticipants = value

    @property
    def rep0(self):
        if self.TestType == "Three Random Repetitions":
            return self.__rep0
        else:
            raise AttributeError("rep1 is not defined for TestType 'Simple Complex'")

    @rep0.setter
    def rep0(self, midi_file_path):
        if self.TestType == "Three Random Repetitions":
            assert midi_file_path.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__rep0 = Pattern(midi_file_path, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("rep1 field is not available for TestType 'Simple Complex'"
                                 "use simple and complex instead or change TestType to 'Three Random Repetitions'")

    @property
    def rep1(self):
        if self.TestType == "Three Random Repetitions":
            return self.__rep1
        else:
            raise AttributeError("rep2 is not defined for TestType 'Simple Complex'")

    @rep1.setter
    def rep1(self, midi_file_path):
        if self.TestType == "Three Random Repetitions":
            assert midi_file_path.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__rep1 = Pattern(midi_file_path, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("rep2 field is not available for TestType 'Simple Complex'"
                                 "use simple and complex instead or change TestType to 'Three Random Repetitions'")

    @property
    def rep2(self):
        if self.TestType == "Three Random Repetitions":
            return self.__rep2
        else:
            raise AttributeError("rep3 is not defined for TestType 'Simple Complex'")

    @rep2.setter
    def rep2(self, midi_file_path):
        if self.TestType == "Three Random Repetitions":
            assert midi_file_path.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__rep2 = Pattern(midi_file_path, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("rep3 field is not available for TestType 'Simple Complex'"
                                 "use simple and complex instead or change TestType to 'Three Random Repetitions'")

    @property
    def simple(self):
        if self.TestType == "Simple Complex":
            return self.__simple
        else:
            raise AttributeError("simple is not defined for TestType 'Three Random Repetitions'")

    @simple.setter
    def simple(self, midi_file_path):
        if self.TestType == "Simple Complex":
            assert midi_file_path.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__simple = Pattern(midi_file_path, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("simple field is not available for TestType 'Three Random Repetitions'"
                                 "use rep1, rep2 and rep3 instead or change TestType to 'Simple Complex'")

    @property
    def complex(self):
        if self.TestType == "Simple Complex":
            return self.__complex
        else:
            raise AttributeError("complex is not defined for TestType 'Three Random Repetitions'")

    @complex.setter
    def complex(self, midi_file_path):
        if self.TestType == "Simple Complex":
            assert midi_file_path.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__complex = Pattern(midi_file_path, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("complex field is not available for TestType 'Three Random Repetitions'"
                                 "use rep1, rep2 and rep3 instead or change TestType to 'Simple Complex'")

    @property
    def three_random_repetitions(self):
        if self.TestType == "Three Random Repetitions":
            return [self.rep0, self.rep1, self.rep2]
        else:
            raise AttributeError("Not Available for TestType 'Simple Complex'")

    @property
    def simple_complex_repetitions(self):
        if self.TestType == "Simple Complex":
            return [self.simple, self.complex]
        else:
            raise AttributeError("Not Available for TestType 'Three Random Repetitions'")

    @property
    def original(self):
        return self.__original

    @original.setter
    def original(self, value):
        assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
        self.__original = Pattern(value, self.style, self.tempo, self.__TestNumber)

    @property
    def style(self):
        return self.__style

    @style.setter
    def style(self, value):
        self.__style = value

    @property
    def tempo(self):
        return self.__tempo

    @tempo.setter
    def tempo(self, value):
        self.__tempo = value

    def piano_rolls(self, show_=False):
        if self.TestType == "Three Random Repetitions":
            tabs = [self.original.piano_roll(), self.rep0.piano_roll(), self.rep1.piano_roll(), self.rep2.piano_roll()]
            title_ = self.FolderPath.split(os.sep)[-1].split(" ")[:2]
            title_ = " ".join(title_)
            titles = [title_, self.rep0.path.split(os.sep)[-1], self.rep1.path.split(os.sep)[-1],
                      self.rep2.path.split(os.sep)[-1]]
            tabs = Tabs(tabs=[Panel(child=tab, title=titles[i]) for i, tab in enumerate(tabs)])
            if show_:
                show(tabs)
            return tabs

        elif self.TestType == "Simple Complex":
            tabs = [self.original.piano_roll(), self.simple.piano_roll(), self.complex.piano_roll()]
            title_ = self.FolderPath.split(os.sep)[-1].split(" ")[:2]
            title_ = " ".join(title_)
            titles = [title_, self.simple.path.split(os.sep)[-1], self.complex.path.split(os.sep)[-1]]
            tabs = Tabs(tabs=[Panel(child=tab, title=titles[i]) for i, tab in enumerate(tabs)])
            if show_:
                show(tabs)
            return tabs
        else:
            raise AttributeError("Not Available for TestType 'Three Random Repetitions'")

    def z_plots(self, show_=False, quantize_all=False, flatten_all=False,
                color_with_vel_values=False, show_balance_evenness_entropy=False, scale_plot=1):
        if _HAS_HVO_SEQUENCE:
            if self.TestType == "Three Random Repetitions":
                tabs = [self.original.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                             color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                             show_balance_evenness_entropy=show_balance_evenness_entropy),
                        self.rep0.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                         color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                         show_balance_evenness_entropy=show_balance_evenness_entropy),
                        self.rep1.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                         color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                         show_balance_evenness_entropy=show_balance_evenness_entropy),
                        self.rep2.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                         color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                         show_balance_evenness_entropy=show_balance_evenness_entropy)]
                titles = [self.short_name, self.rep0.short_name,
                          self.rep1.short_name,
                          self.rep2.short_name]
                tabs = Tabs(tabs=[Panel(child=tab, title=titles[i]) for i, tab in enumerate(tabs)])
                if show_:
                    show(tabs)
                return tabs
            elif self.TestType == "Simple Complex":
                tabs = [self.original.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                             color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                             show_balance_evenness_entropy=show_balance_evenness_entropy),
                        self.simple.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                           color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                           show_balance_evenness_entropy=show_balance_evenness_entropy),
                        self.complex.z_plot(quantize_all=quantize_all, flatten_all=flatten_all,
                                            color_with_vel_values=color_with_vel_values, scale_plot=scale_plot,
                                            show_balance_evenness_entropy=show_balance_evenness_entropy)]
                titles = [self.short_name, self.simple.short_name, self.complex.short_name]
                tabs = Tabs(tabs=[Panel(child=tab, title=titles[i]) for i, tab in enumerate(tabs)])
                if show_:
                    show(tabs)
                return tabs

        else:
            raise AttributeError("Not Available for TestType 'Three Random Repetitions'")

    def audios(self, soundfont=None, auto_play=False):
        if self.TestType == "Three Random Repetitions":
            audios_ = [self.original.audio(soundfont=soundfont, auto_play=auto_play),
                    self.rep0.audio(soundfont=soundfont, auto_play=auto_play),
                    self.rep1.audio(soundfont=soundfont, auto_play=auto_play),
                    self.rep2.audio(soundfont=soundfont, auto_play=auto_play)]
            audios = {
                f"{self.short_name} "+self.original.short_name: audios_[0],
                f"{self.short_name} "+self.rep0.short_name: audios_[1],
                f"{self.short_name} "+self.rep1.short_name: audios_[2],
                f"{self.short_name} "+self.rep2.short_name: audios_[3]
            }
            return audios
        elif self.TestType == "Simple Complex":
            audios_ = [self.original.audio(soundfont=soundfont, auto_play=auto_play),
                    self.simple.audio(soundfont=soundfont, auto_play=auto_play),
                    self.complex.audio(soundfont=soundfont, auto_play=auto_play)]
            audios = {
                f"{self.short_name} "+self.original.short_name: audios_[0],
                f"{self.short_name} "+self.simple.short_name: audios_[1],
                f"{self.short_name} "+self.complex.short_name: audios_[2]
            }
            return audios
        else:
            return None

    def calculate_intra_dualization_edit_distances(self, normalize_by_union=False):
        edit_distances = []
        combs = None
        if self.TestType == "Three Random Repetitions":
            combs = combinations([self.rep0, self.rep1, self.rep2], 2)
        elif self.TestType == "Simple Complex":
            combs = combinations([self.simple, self.complex], 2)
        for comb in combs:
            edit_distances.append(
                round(comb[0].calculate_edit_distance_with(comb[1], normalize_by_union=normalize_by_union), 2))
        return edit_distances

    def calculate_inter_dualization_edit_distances(self, other_ParticipantDualizations,
                                                   throw_error_if_not_same_test_type=True, normalize_by_union=False):
        edit_distances = []
        if self.TestType != other_ParticipantDualizations.TestType:
            if throw_error_if_not_same_test_type:
                raise AttributeError("Patterns must have the same TestType")
            else:
                return None

        source = None
        target = None

        if other_ParticipantDualizations.TestType == "Three Random Repetitions":
            source = [self.rep0, self.rep1, self.rep2]
            target = [other_ParticipantDualizations.rep0, other_ParticipantDualizations.rep1,
                      other_ParticipantDualizations.rep2]
            for s in source:
                for t in target:
                    edit_distances.append(s.calculate_edit_distance_with(t))
        elif other_ParticipantDualizations.TestType == "Simple Complex":
             edit_distances = {
                "Simple": self.simple.calculate_edit_distance_with(other_ParticipantDualizations.simple,
                                                                   normalize_by_union=normalize_by_union),
                "Complex": self.complex.calculate_edit_distance_with(other_ParticipantDualizations.complex,
                                                                     normalize_by_union=normalize_by_union)
            }


        return edit_distances
    def calculate_dualization_to_original_edit_distances(self, normalize_by_union=False):
        edit_distances = []
        if self.TestType == "Three Random Repetitions":
            edit_distances.append(self.original.calculate_edit_distance_with(self.rep0,
                                                                             normalize_by_union=normalize_by_union))
            edit_distances.append(self.original.calculate_edit_distance_with(self.rep1,
                                                                             normalize_by_union=normalize_by_union))
            edit_distances.append(self.original.calculate_edit_distance_with(self.rep2,
                                                                             normalize_by_union=normalize_by_union))
        elif self.TestType == "Simple Complex":
            edit_distances.append(self.original.calculate_edit_distance_with(self.simple,
                                                                             normalize_by_union=normalize_by_union))
            edit_distances.append(self.original.calculate_edit_distance_with(self.complex,
                                                                             normalize_by_union=normalize_by_union))
        return edit_distances

    def calculate_intra_dualization_jaccard_similarities(self):
        jaccard_similarities = []
        combs = None
        if self.TestType == "Three Random Repetitions":
            combs = combinations([self.rep0, self.rep1, self.rep2], 2)
        elif self.TestType == "Simple Complex":
            combs = combinations([self.simple, self.complex], 2)
        for comb in combs:
            jaccard_similarities.append(comb[0].calculate_jaccard_similarity_with(comb[1]))
        return jaccard_similarities

    def calculate_inter_dualization_jaccard_similarities(self, other_ParticipantDualizations,
                                                        throw_error_if_not_same_test_type=True):
        jaccard_similarities = []
        if self.TestType != other_ParticipantDualizations.TestType:
            if throw_error_if_not_same_test_type:
                raise AttributeError("Patterns must have the same TestType")
            else:
                return None

        source = None
        target = None


        if other_ParticipantDualizations.TestType == "Three Random Repetitions":
            source = [self.rep0, self.rep1, self.rep2]
            target = [other_ParticipantDualizations.rep0, other_ParticipantDualizations.rep1,
                      other_ParticipantDualizations.rep2]
            for s in source:
                for t in target:
                    jaccard_similarities.append(s.calculate_jaccard_similarity_with(t))
        elif other_ParticipantDualizations.TestType == "Simple Complex":
            jaccard_similarities = {
                "Simple": self.simple.calculate_jaccard_similarity_with(other_ParticipantDualizations.simple),
                "Complex": self.complex.calculate_jaccard_similarity_with(other_ParticipantDualizations.complex)
            }
        return jaccard_similarities

    def calculate_dualization_to_original_jaccard_similarities(self):
        jaccard_similarities = []
        if self.TestType == "Three Random Repetitions":
            jaccard_similarities.append(self.original.calculate_jaccard_similarity_with(self.rep0))
            jaccard_similarities.append(self.original.calculate_jaccard_similarity_with(self.rep1))
            jaccard_similarities.append(self.original.calculate_jaccard_similarity_with(self.rep2))
        elif self.TestType == "Simple Complex":
            jaccard_similarities.append(self.original.calculate_jaccard_similarity_with(self.simple))
            jaccard_similarities.append(self.original.calculate_jaccard_similarity_with(self.complex))
        return jaccard_similarities

    def collect_step_densities(self, normalize_by_original=False):
        original_density = self.original.calculate_step_density() if normalize_by_original else 1

        densities = None
        if self.TestType == "Three Random Repetitions":
            densities = []
            densities.append(self.rep0.calculate_step_density() / original_density)
            densities.append(self.rep1.calculate_step_density() / original_density)
            densities.append(self.rep2.calculate_step_density() / original_density)

        elif self.TestType == "Simple Complex":
            densities = {
                "Simple": self.simple.calculate_step_density() / original_density,
                "Complex": self.complex.calculate_step_density() / original_density
            }

        return densities

    def calculate_intra_dualization_cohens_kappa(self):
        cohens_kappas = []
        combs = None
        if self.TestType == "Three Random Repetitions":
            combs = combinations([self.rep0, self.rep1, self.rep2], 2)
        elif self.TestType == "Simple Complex":
            combs = combinations([self.simple, self.complex], 2)
        for comb in combs:
            cohens_kappas.append(comb[0].calculate_cohens_kappa_with(comb[1]))
        return cohens_kappas

    def calculate_inter_dualization_cohens_kappa(self, other_ParticipantDualizations,
                                                        throw_error_if_not_same_test_type=True):
        """
        Calculates the Cohen's Kappa between the dualizations of two participants.
        :param other_ParticipantDualizations: The other participant's dualizations
        :param throw_error_if_not_same_test_type: If True, will throw an error if the two participants have different
        test types (i.e. comparing a simple-complex participant to a three-random-repetitions participant will not

        NOTE: AT THIS POINT, COMPARING SIMPLE-COMPLEX TO THREE-RANDOM-REPETITIONs AND VICE VERSA IS NOT SUPPORTED.
        """
        cohens_kappas = []
        if self.TestType != other_ParticipantDualizations.TestType:
            if throw_error_if_not_same_test_type:
                raise AttributeError("Patterns must have the same TestType")
            else:
                return None

        source = None
        target = None
        if other_ParticipantDualizations.TestType == "Three Random Repetitions":
            source = [self.rep0, self.rep1, self.rep2]
            target = [other_ParticipantDualizations.rep0, other_ParticipantDualizations.rep1,
                      other_ParticipantDualizations.rep2]
            for s in source:
                for t in target:
                    cohens_kappas.append(s.calculate_cohens_kappa_with(t))
        elif other_ParticipantDualizations.TestType == "Simple Complex":
            cohens_kappas = {
                "Simple": self.simple.calculate_cohens_kappa_with(other_ParticipantDualizations.simple),
                "Complex": self.complex.calculate_cohens_kappa_with(other_ParticipantDualizations.complex)
            }
        return cohens_kappas
class Pattern:
    def __init__(self, midi_file_path, style, tempo, test_number):
        self.__path = midi_file_path
        self.__hvo_sequence = None
        self.__style = style
        self.__tempo = tempo
        self.__test_number = test_number
        self.__repetition = None
        splitted_path = midi_file_path.split(os.sep)[-1].split("_")
        if len(splitted_path) > 2:
            self.repetition = " ".join(splitted_path[2:]).capitalize().replace(".mid", "")
        else:
            self.repetition = " ".join(splitted_path).capitalize().replace(".mid", "")

    def __repr__(self):
        return f"Pattern Midi Path: {self.path}"

    @property
    def path(self):
        return self.__path

    @property
    def short_name(self):
        n_ = self.path.split(os.sep)[-1].replace("Participant_", "P").replace("repetition", "rep").replace(".mid", "")
        return n_

    @property
    def repetition(self):
        return self.__repetition

    @repetition.setter
    def repetition(self, value):
        assert isinstance(value, str), "repetition must be a string"
        assert value in ["Repetition 0", "Repetition 1", "Repetition 2", "Simple", "Complex", "Original"], \
            "repetition must be one of the following: \n " \
            "   'Repetition 0', 'Repetition 1', 'Repetition 2', 'Simple', 'Complex', 'Original'"
        self.__repetition = value

    @property
    def hvo_sequence(self):
        if _HAS_HVO_SEQUENCE:
            if self.__hvo_sequence is None:
                dmap = ROLAND_REDUCED_MAPPING_HEATMAPS if "original" in self.path else DUALIZATION_ROLAND_HAND_DRUM
                self.__hvo_sequence = midi_to_hvo_sequence(filename=self.__path,
                                                           drum_mapping=dmap,
                                                           beat_division_factors=[4])
                self.__hvo_sequence.adjust_length(32)
                self.__hvo_sequence.metadata.update({
                    "Style": self.__style,
                    "Tempo": self.__tempo,
                    "TestNumber": f"{int(self.__test_number):03d}",
                    "Repetition": self.__repetition
                })
            return self.__hvo_sequence
        else:
            print("hvo_sequence is not available. Please install requirements")
            return None

    def randomize_hits(self):
        if _HAS_HVO_SEQUENCE:
            if self.__hvo_sequence is None:
                self.hvo_sequence
            self.__hvo_sequence.random(self.hvo_sequence.number_of_steps)
        else:
            print("hvo_sequence is not available. Please install requirements")
    @property
    def pretty_midi(self):
        if _HAS_PRETTY_MIDI:
            return pretty_midi.PrettyMIDI(self.path)
        else:
            print("pretty_midi is not available. Please install requirements")
            return None

    @property
    def note_sequence(self):
        if _HAS_NOTE_SEQ:
            return note_seq.midi_file_to_note_sequence(self.path)
        else:
            print("note_sequence is not available. Please install requirements")
            return None

    def piano_roll(self, show=False, width=800, height=400):
        if _HAS_HVO_SEQUENCE:
            title = " ".join(self.path.split(os.sep)[-2].split(" ")[:2]) + " -- " + self.path.split(os.sep)[-1].replace(".mid", "")
            fig = self.hvo_sequence.to_html_plot(show_figure=show, filename=title, width=width, height=height)
            return fig
        else:
            print("piano_roll is not available. Please install requirements for HVO_Sequence")
            return None

    def z_plot(self, quantize=False, flatten=False, scale_plot=1,
               compare_with_other_Pattern=None, quantize_other=False, flatten_other=False,
               line_color="black", show_figure=False, filename=None,
               quantize_all=False, flatten_all=False, color_with_vel_values=True, show_balance_evenness_entropy=False,):
        plot_width = int(500 * scale_plot)
        plot_height = int(500 * scale_plot)
        identifier = self.path.split(os.sep)[-1].replace(".mid", "") + \
                     f" {self.__test_number}"
        if compare_with_other_Pattern is not None:
            assert isinstance(compare_with_other_Pattern, Pattern), "compare_with_other_Pattern must be a Pattern object"
            compare_with_other_hvo_sequence = compare_with_other_Pattern.hvo_sequence
            other_identifier = compare_with_other_Pattern.path.split(os.sep)[-1].replace(".mid", "") + \
                               f" {compare_with_other_Pattern.__test_number}"
        else:
            compare_with_other_hvo_sequence = None
            other_identifier = None

        if _HAS_HVO_SEQUENCE:

            fig = self.hvo_sequence.get_argand_plot(
                title="",
                identifier=identifier,
                quantize=quantize,
                flatten=flatten,
                compare_with_other_hvo_sequence=compare_with_other_hvo_sequence,
                other_identifier=other_identifier,
                quantize_other=quantize_other,
                flatten_other=flatten_other,
                plot_width=plot_width,
                plot_height=plot_height,
                line_color=line_color,
                show_figure=show_figure,
                filename=filename,
                quantize_all=quantize_all,
                flatten_all=flatten_all,
                color_with_vel_values=color_with_vel_values,
                show_balance_evenness_entropy=show_balance_evenness_entropy,
            )

            return fig

    def audio(self, soundfont=None, auto_play=True):
        if _HAS_HVO_SEQUENCE:
            if soundfont is None:
                soundfont = "hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"
            audio = self.hvo_sequence.synthesize(sf_path=soundfont)
            if auto_play:
                if _HAS_AUDIO_PLAYER:
                    display(Audio(audio, rate=44100))
                else:
                    print("auto_play is not available. Please install requirements for IPython.display.Audio")
            return audio
        else:
            print("synthesize is not available. Please install requirements for HVO_Sequence")
            return None

    def calculate_edit_distance_with(self, otherPattern, normalize_by_union=False):
        if normalize_by_union:
            a = self.hvo_sequence.flatten_voices(reduce_dim=True)[:, 0]
            b = otherPattern.hvo_sequence.flatten_voices(reduce_dim=True)[:, 0]
            set_a = set([i for i, x in enumerate(a) if x == 1])
            set_b = set([i for i, x in enumerate(b) if x == 1])
            union = len(set_a.union(set_b))
            return self.hvo_sequence.calculate_edit_distance_with(otherPattern.hvo_sequence, reduce_dim=True) / union
        else:
            return self.hvo_sequence.calculate_edit_distance_with(otherPattern.hvo_sequence, reduce_dim=True)

    def calculate_jaccard_similarity_with(self, otherPattern):
        return round(self.hvo_sequence.calculate_jaccard_similarity_with(otherPattern.hvo_sequence, reduce_dim=True), 2)

    def calculate_step_density(self):
        return self.hvo_sequence.get_total_step_density()

    def calculate_cohens_kappa_with(self, otherPattern):
        if _HAS_HVO_SEQUENCE:
            pattern_a = self.hvo_sequence.flatten_voices(reduce_dim=True)[:, 0]
            pattern_b = otherPattern.hvo_sequence.flatten_voices(reduce_dim=True)[:, 0]
            return cohen_kappa_score(pattern_a, pattern_b)


if __name__ == "__main__":
    # summary = collect_data("midi_files", "./midi_files/summary.csv")

    # print(summary)
    dataset = DualizationDatasetAPI(midi_folder="midi_files")

    # print(dataset.summary_dict)

    dataset.save_summary("midi_files/summary2.csv")

    new_dataset = dataset.get_subset_matching_styles("jazz", hard_match=True)
    print(new_dataset.summary_dataframe["Style"])
    new_dataset.save_summary("midi_files/summary2jazz.csv")

    # dataset.get_participants_attempted_test_number("100")
    # dataset.get_test_numbers()
    #
    # dataset.get_folder_path_for_test_number("100")
    #
    # dataset.get_tested_drum_pattern_path("100")
    #
    # dualizations = dataset.get_participant_dualizations_for_test_number(test_number=100, participant=1)
    #
    # dataset.get_tested_drum_pattern_path("100")
