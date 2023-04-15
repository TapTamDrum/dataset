import numpy as np
from hvo_sequence.custom_dtypes import Tempo, Time_Signature
import math
import scipy.signal

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

import warnings
from logging import getLogger

logger = getLogger('hvo_sequence/utils')
logger.setLevel("DEBUG")


def find_nearest(array, query):
    """
    Finds the closest entry in array to query. array must be sorted!
    @param array:                   a sorted array to search within
    @param query:                   value to find the nearest to
    @return index, array[index]:    the index in array closest to query, and the actual value
    """
    index = (np.abs(array-query)).argmin()
    return index, array[index]


def is_power_of_two(n):
    """
    Checks if a value is a power of two
    @param n:                               # value to check (must be int or float - otherwise assert error)
    @return:                                # True if a power of two, else false
    """
    if n is None:
        return False

    assert (isinstance(n, int) or isinstance(n, float)), "The value to check must be either int or float"

    if (isinstance(n, float) and n.is_integer()) or isinstance(n, int):
        # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        n = int(n)
        return (n & (n - 1) == 0) and n != 0
    else:
        return False


def find_pitch_and_tag(pitch_query, drum_mapping):
    """
    checks to which drum group the pitch belongs,
    then returns the index for group and the first pitch in group

    @param pitch_query:                 pitch_query for which the corresponding drum voice group is found
    @param drum_mapping:                dict of {'Drum Voice Tag': [midi numbers]}
    @return tuple of mapped_pitch,      index of the drum voice group the pitch belongs to
            instrument_tag,:            the first pitch in the corresponding drum group
                                        Note: Returns (None, None) if no matching group is found
            pitch_class_ix:             The ith group to which th pitch_query belongs
    """

    for ix, instrument_tag in enumerate(drum_mapping.keys()):
        if pitch_query in drum_mapping[instrument_tag]:
            mapped_pitch = drum_mapping[instrument_tag][0]
            return mapped_pitch, instrument_tag, ix

    # If pitch_query isn't in the pitch_class_list, return None, None, None
    return None, None, None


def create_grid_for_n_bars(n_bars, time_signature, tempo):

    # Creates n bars of grid lines according to the tempo and time_signature and the
    # requested beat_division_factors
    # ALso, returns the position of the beginning of the next bar/measure

    assert isinstance(time_signature, Time_Signature), "time_signature should be an instance of Time_Signature class"
    assert time_signature.is_ready_to_use, "There are missing fields in time_signature instance"
    assert isinstance(tempo, Tempo), "tempo should be an instance of Tempo class"
    assert tempo.is_ready_to_use, "There are missing fields in the tempo instance"

    # Calculate beat duration (beat defined based on signature denominator) --> not the perceived beat
    beat_dur = (60.0 / tempo.qpm) * 4.0 / time_signature.denominator

    # Calculate the number of beats
    n_beats = n_bars * time_signature.numerator

    # Empty grid
    grid_lines = np.array([])

    for ix, beat_div_factor in enumerate(time_signature.beat_division_factors):
        grid_lines = np.append(grid_lines, np.arange(n_beats * beat_div_factor) * beat_dur / beat_div_factor)

    beginning_of_next_bar = n_beats*beat_dur
    return np.unique(grid_lines), beginning_of_next_bar


def cosine_similarity(hvo_seq_a, hvo_seq_b):
    assert hvo_seq_a.hvo.shape[-1] == hvo_seq_b.hvo.shape[-1], "the two sequences must have the same last dimension"
    assert len(hvo_seq_a.tempos) == 1 and len(hvo_seq_a.time_signatures) == 1, \
        "Input A Currently doesn't support multiple tempos or time_signatures"
    assert len(hvo_seq_b.tempos) == 1 and len(hvo_seq_b.time_signatures) == 1, \
        "Input B Currently doesn't support multiple tempos or time_signatures"

    # Ensure a and b have same length by Padding the shorter sequence to match the longer one
    max_len = max(hvo_seq_a.hvo.shape[0], hvo_seq_b.hvo.shape[0])


    hvo_a = hvo_seq_a.hvo.flatten()
    hvo_b = hvo_seq_b.hvo.flatten()

    # Calculate cosine similarity

    return 1-np.dot(hvo_a, hvo_b)/(np.linalg.norm(hvo_a)*np.linalg.norm(hvo_b))


def cosine_distance(hvo_seq_a, hvo_seq_b,  ):
    return 1-cosine_similarity(hvo_seq_a, hvo_seq_b)


def fuzzy_Hamming_distance(velocity_grooveA, utiming_grooveA,
                           velocity_grooveB, utiming_grooveB,
                           beat_weighting=False):
    # Get fuzzy Hamming distance as velocity weighted Hamming distance, but with 1 metrical distance lookahead/back
    # and microtiming weighting
    # Microtiming must be in ms with nan whenever theres no hit
    assert velocity_grooveA.shape[0] == 32 and \
           velocity_grooveB.shape[0] == 32, "Currently only supports calculation on 2 bar " \
                                                    "loops in 4/4 and 16th note quantization"

    a = velocity_grooveA
    a_timing = utiming_grooveA
    b = velocity_grooveB
    b_timing = utiming_grooveB

    if beat_weighting is True:
        a = _weight_groove(a)
        b = _weight_groove(b)

    timing_difference = np.nan_to_num(a_timing - b_timing)

    x = np.zeros(a.shape)
    tempo = 120.0
    steptime_ms = 60.0 * 1000 / tempo / 4 # semiquaver step time in ms

    difference_weight = timing_difference / 125.
    difference_weight = 1+np.absolute(difference_weight)
    single_difference_weight = 400

    for j in range(a.shape[-1]):
        for i in range(31):
            if a[i,j] != 0.0 and b[i,j] != 0.0:
                x[i,j] = (a[i,j] - b[i,j]) * (difference_weight[i,j])
            elif a[i,j] != 0.0 and b[i,j] == 0.0:
                if b[(i+1) % 32, j] != 0.0 and a[(i+1) % 32, j] == 0.0:
                    single_difference = np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i+1)%32,j]) + steptime_ms
                    if single_difference < 125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i+1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif b[(i-1)%32,j] != 0.0 and a[(i-1)%32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i-1)%32,j]) - steptime_ms

                    if single_difference > -125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i-1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]
                else:
                    x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

            elif a[i,j] == 0.0 and b[i,j] != 0.0:
                if b[(i + 1) % 32, j] != 0.0 and a[(i + 1) % 32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i+1)%32,j]) + steptime_ms
                    if single_difference < 125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i+1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif b[(i-1)%32,j] != 0.0 and a[(i-1)%32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i-1)%32,j]) - steptime_ms
                    if single_difference > -125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i-1)%32,j]) * single_difference_weight

                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                else: # if no nearby onsets, need to count difference between onset and 0 value.
                    x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

    fuzzy_distance = math.sqrt(np.dot(x.flatten(), x.flatten().T))
    return fuzzy_distance

def _weight_groove(_velocity_groove):
    # Metrical awareness profile weighting for hamming distance.
    # The rhythms in each beat of a bar have different significance based on GTTM

    # Repeat the awareness profile for each voice
    beat_awareness_weighting = np.array([1, 1, 1, 1,
                                         0.27, 0.27, 0.27, 0.27,
                                         0.22, 0.22, 0.22, 0.22,
                                         0.16, 0.16, 0.16, 0.16])

    # Match size of weighting factors and velocity groove
    if _velocity_groove.shape[0] > beat_awareness_weighting.shape[0]:
        pad_size = _velocity_groove.shape[0] - beat_awareness_weighting.shape[0]
        beat_awareness_weighting = np.pad(beat_awareness_weighting, (0, pad_size), mode='wrap').reshape(-1, 1)

    beat_awareness_weighting = beat_awareness_weighting[:_velocity_groove.shape[0], :]

    # Apply weight
    weighted_groove = _velocity_groove * beat_awareness_weighting

    return weighted_groove


def _reduce_part(part, metrical_profile):
    length = part.shape[0]
    for i in range(length):
        if part[i] <= 0.4:
            part[i] = 0
    for i in range(length):
        if part[i] != 0.:  # hit detected - must be figural or density transform - on pulse i.
            for k in range(-3, i):  # iterate through all previous events up to i.
                if part[k] != 0. and metrical_profile[k] < metrical_profile[i]:
                    # if there is a preceding event in a weaker pulse k (this is to be removed)

                    # groove[k,0] then becomes k, and can either be density of figural
                    previous_event_index = 0
                    for l in range(0, k):  # find strongest level pulse before k, with no events between m and k
                        if part[l] != 0.:  # find an event if there is one
                            previous_event_index = l
                        else:
                            previous_event_index = 0
                    m = max(metrical_profile[
                            previous_event_index:k])  # find the strongest level between previous event index and k.
                    # search for largest value in salience profile list between range l+1 and k-1. this is m.
                    if m <= k:  # density if m not stronger than k
                        part[k] = 0  # to remove a density transform just remove the note
                    if m > k:  # figural transform
                        part[m] = part[k]  # need to shift note forward - k to m.
                        part[k] = 0  # need to shift note forward - k to m.
        if part[i] == 0:
            for k in range(-3, i):
                if part[k] != 0. and metrical_profile[k] < metrical_profile[i]:  # syncopation detected
                    part[i] = part[k]
                    part[k] = 0.0
    return part

def _get_2bar_segments(part, steps_in_measure):
    """
    returns a list of np.array each element of which is a 2bar part
    Pads and replicates if len adjustment is needed
    """
    part = part.reshape(-1, 1) if part.ndim == 1 else part  # reshape to (n_steps, 1)

    # first make_sure_length_is_multiple of 16, if not append zero arrays
    if part.shape[0] % steps_in_measure != 0:
        pad_size = int(np.ceil(part.shape[0]/steps_in_measure)*steps_in_measure - part.shape[0])
        part = np.pad(part, ((0, pad_size), (0, 0)), mode="constant")

    # match length to multiple 2 bars (if shorter repeat last bar)
    if part.shape[0] % (2 * steps_in_measure) != 0:
        part = np.append(part, part[-steps_in_measure:, :], axis=0)

    two_bar_segments = np.split(part, part.shape[0] // (steps_in_measure * 2))
    return two_bar_segments

def _get_kick_and_snare_syncopations(low, mid, high, i, metrical_profile, steps_in_measure=16):
    """
    Makes sure that the pattern is fitted and splitted into two bar measures
    then averages each 2bar segment's syncopation (calculated via get_monophonic_syncopation_for_2bar)

    :param part:
    :param metrical_profile:
    :param steps_in_measure:
    :return:
    """
    low_2bar_segs = _get_2bar_segments(low, steps_in_measure=steps_in_measure)
    mid_2bar_segs = _get_2bar_segments(mid, steps_in_measure=steps_in_measure)
    high_2bar_segs = _get_2bar_segments(high, steps_in_measure=steps_in_measure)

    kick_syncs_per_two_bar_segments = np.array([])
    snare_syncs_per_two_bar_segments = np.array([])

    for seg_ix, _ in enumerate(low_2bar_segs):
        kick_syncs_per_two_bar_segments = np.append(
            kick_syncs_per_two_bar_segments,
            _get_kick_syncopation_for_2bar(
                low_2bar_segs[seg_ix],
                mid_2bar_segs[seg_ix],
                high_2bar_segs[seg_ix],
                i,
                metrical_profile)
        )
        snare_syncs_per_two_bar_segments = np.append(
            snare_syncs_per_two_bar_segments,
            _get_snare_syncopation_for_2bar(
                low_2bar_segs[seg_ix],
                mid_2bar_segs[seg_ix],
                high_2bar_segs[seg_ix],
                i,
                metrical_profile)
        )

    return kick_syncs_per_two_bar_segments.mean(), snare_syncs_per_two_bar_segments.mean()


def _get_kick_syncopation_for_2bar(low, mid, high, i, metrical_profile):
    # Find instances  when kick syncopates against hi hat/snare on the beat.
    # For use in polyphonic syncopation feature

    kick_syncopation = 0
    k = 0
    next_hit = ""
    if low[i] == 1 and low[(i + 1) % 32] != 1 and low[(i + 2) % 32] != 1:
        for j in i + 1, i + 2, i + 3, i + 4:  # look one and two steps ahead only - account for semiquaver and quaver sync
            if mid[(j % 32)] == 1 and high[(j % 32)] != 1:
                next_hit = "Mid"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and mid[(j % 32)] != 1:
                next_hit = "High"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and mid[(j % 32)] == 1:
                next_hit = "MidAndHigh"
                k = j % 32
                break
            # if both next two are 0 - next hit == rest. get level of the higher level rest
        if mid[(i + 1) % 32] + mid[(i + 2) % 32] == 0.0 and high[(i + 1) % 32] + [(i + 2) % 32] == 0.0:
            next_hit = "None"

        if next_hit == "MidAndHigh":
            if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                difference = metrical_profile[k] - metrical_profile[i]
                kick_syncopation = difference + 2
        elif next_hit == "Mid":
            if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                difference = metrical_profile[k] - metrical_profile[i]
                kick_syncopation = difference + 2
        elif next_hit == "High":
            if metrical_profile[k] >= metrical_profile[i]:
                difference = metrical_profile[k] - metrical_profile[i]
                kick_syncopation = difference + 5
        elif next_hit == "None":
            if metrical_profile[k] > metrical_profile[i]:
                difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[
                    i]
                kick_syncopation = difference + 6  # if rest on a stronger beat - one stream sync, high sync value
    return kick_syncopation


def _get_snare_syncopation_for_2bar(low, mid, high, i, metrical_profile):
    # Find instances  when snare syncopates against hi hat/kick on the beat
    # For use in polyphonic syncopation feature

    snare_syncopation = 0
    next_hit = ""
    k = 0
    if mid[i] == 1 and mid[(i + 1) % 32] != 1 and mid[(i + 2) % 32] != 1:
        for j in i + 1, i + 2, i + 3, i + 4:  # look one and 2 steps ahead only
            if low[(j % 32)] == 1 and high[(j % 32)] != 1:
                next_hit = "Low"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and low[(j % 32)] != 1:
                next_hit = "High"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and low[(j % 32)] == 1:
                next_hit = "LowAndHigh"
                k = j % 32
                break
        if low[(i + 1) % 32] + low[(i + 2) % 32] == 0.0 and high[(i + 1) % 32] + [(i + 2) % 32] == 0.0:
            next_hit = "None"

        if next_hit == "LowAndHigh":
            if metrical_profile[k] >= metrical_profile[i]:
                difference = metrical_profile[k] - metrical_profile[i]
                snare_syncopation = difference + 1  # may need to make this back to 1?)
        elif next_hit == "Low":
            if metrical_profile[k] >= metrical_profile[i]:
                difference = metrical_profile[k] - metrical_profile[i]
                snare_syncopation = difference + 1
        elif next_hit == "High":
            if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                difference = metrical_profile[k] - metrical_profile[i]
                snare_syncopation = difference + 5
        elif next_hit == "None":
            if metrical_profile[k] > metrical_profile[i]:
                difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[
                    i]
                snare_syncopation = difference + 6  # if rest on a stronger beat - one stream sync, high sync value
    return snare_syncopation


def get_monophonic_syncopation(part, metrical_profile, steps_in_measure=16):
    """
    Makes sure that the pattern is fitted and splitted into two bar measures
    then averages each 2bar segment's syncopation (calculated via get_monophonic_syncopation_for_2bar)

    :param part:
    :param metrical_profile:
    :param steps_in_measure:
    :return:
    """
    two_bar_segments = _get_2bar_segments(part, steps_in_measure)
    syncs_per_two_bar_segments = np.array([])
    for segment in two_bar_segments:
        syncs_per_two_bar_segments = np.append(
            syncs_per_two_bar_segments,
            get_monophonic_syncopation_for_2bar(segment, metrical_profile)
        )

    return syncs_per_two_bar_segments.mean()

# todo: adapt to length variations (other than 2 bars)
def get_monophonic_syncopation_for_2bar(_2bar_part, metrical_profile):
    """
    Calculates monophonic syncopation levels for a 2 bar segment in 4-4 meter and 16th note quantization
    :param _2bar_part:
    :param metrical_profile:
    :return:
    """
    if all(np.isnan(_2bar_part)):
        return 0

    max_syncopation = 30.0
    syncopation = 0.0

    for i in range(len(_2bar_part)):
        if _2bar_part[i] != 0:
            if _2bar_part[(i + 1) % 32] == 0.0 and metrical_profile[(i + 1) % 32] > metrical_profile[i]:
                syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 1) % 32] - metrical_profile[i])))  # * part[i])) #todo: velocity here?

            elif _2bar_part[(i + 2) % 32] == 0.0 and metrical_profile[(i + 2) % 32] > metrical_profile[i]:
                syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 2) % 32] - metrical_profile[i])))  # * part[i]))

    return syncopation / max_syncopation


def get_weak_to_strong_ratio(velocity_groove):
    """
    returns the ratio of total weak onsets divided by all strong onsets
    strong onsets are onsets that occur on beat positions and weak onsets are the other ones
    """

    part = velocity_groove

    weak_hit_count = 0.0
    strong_hit_count = 0.0

    strong_positions = [0, 4, 8, 12, 16, 20, 24, 28]
    weak_positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17,
                      18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]

    hits_count = np.count_nonzero(part)
    hit_indexes = np.nonzero(part)
    for i in range(hits_count):
        if len(hit_indexes) > 1:
            index = hit_indexes[0][i], hit_indexes[1][i]
        else:
            index = [hit_indexes[0][i]]
        if (index[0] % 32) in strong_positions:
            strong_hit_count += part[index]
        if (index[0] % 32) in weak_positions:
            weak_hit_count += part[index]

    if strong_hit_count>0:
        return weak_hit_count/strong_hit_count
    else:
        return 0


def _getmicrotiming_event_profile_1bar(microtiming_matrix, kick_ix, snare_ix, chat_ix, threshold):
    # Get profile of microtiming events for use in pushness/laidbackness/ontopness features
    # This profile represents the presence of specific timing events at certain positions in the pattern
    # Microtiming events fall within the following categories:
    #   Kick timing deviation - before/after metronome, before/after hihat, beats 1 and 3
    #   Snare timing deviation - before/after metronome, before/after hihat, beats 2 and 4
    # As such for one bar the profile contains 16 values.
    # The profile uses binary values - it only measures the presence of timing events, and the style features are
    # then calculated based on the number of events present that correspond to a certain timing feel.

    timing_to_grid_profile = np.zeros([8])
    timing_to_cymbal_profile = np.zeros([8])

    kick_timing_1 = microtiming_matrix[0, kick_ix]
    hihat_timing_1 = microtiming_matrix[0, chat_ix]
    snare_timing2 = microtiming_matrix[4, snare_ix]
    hihat_timing_2 = microtiming_matrix[4, chat_ix]
    kick_timing_3 = microtiming_matrix[8, kick_ix]
    hihat_timing_3 = microtiming_matrix[8, chat_ix]
    snare_timing4 = microtiming_matrix[12, snare_ix]
    hihat_timing_4 = microtiming_matrix[12, chat_ix]

    if kick_timing_1 > threshold:
        timing_to_grid_profile[0] = 1
    if kick_timing_1 < -threshold:
        timing_to_grid_profile[1] = 1
    if snare_timing2 > threshold:
        timing_to_grid_profile[2] = 1
    if snare_timing2 < -threshold:
        timing_to_grid_profile[3] = 1

    if kick_timing_3 > threshold:
        timing_to_grid_profile[4] = 1
    if kick_timing_3 < -threshold:
        timing_to_grid_profile[5] = 1
    if snare_timing4 > threshold:
        timing_to_grid_profile[6] = 1
    if snare_timing4 < -threshold:
        timing_to_grid_profile[7] = 1

    if kick_timing_1 > hihat_timing_1 + threshold:
        timing_to_cymbal_profile[0] = 1
    if kick_timing_1 < hihat_timing_1 - threshold:
        timing_to_cymbal_profile[1] = 1
    if snare_timing2 > hihat_timing_2 + threshold:
        timing_to_cymbal_profile[2] = 1
    if snare_timing2 < hihat_timing_2 - threshold:
        timing_to_cymbal_profile[3] = 1

    if kick_timing_3 > hihat_timing_3 + threshold:
        timing_to_cymbal_profile[4] = 1
    if kick_timing_3 < hihat_timing_3 - threshold:
        timing_to_cymbal_profile[5] = 1
    if snare_timing4 > hihat_timing_4 + threshold:
        timing_to_cymbal_profile[6] = 1
    if snare_timing4 < hihat_timing_4 - threshold:
        timing_to_cymbal_profile[7] = 1

    microtiming_event_profile_1bar = np.clip(timing_to_grid_profile + timing_to_cymbal_profile, 0, 1)

    return microtiming_event_profile_1bar


#   -------------------------------------------------------------
#   Utils for computing the MSO::Multiband Synthesized Onsets
#   -------------------------------------------------------------

def cq_matrix(n_bins_per_octave, n_bins, f_min, n_fft, sr):
    """
    Constant-Q filterbank frequencies
    Based on https://github.com/mcartwright/dafx2018_adt/blob/master/large_vocab_adt_dafx2018/features.py
    @param n_bins_per_octave: int
    @param n_bins: int
    @param f_min: float
    @param n_fft: int
    @param sr: int
    @return c_mat: matrix
    @return: f_cq: list (triangular filters center frequencies)
    """
    # note range goes from -1 to bpo*num_oct for boundary issues
    f_cq = f_min * 2 ** ((np.arange(-1, n_bins + 1)) / n_bins_per_octave)  # center frequencies
    # centers in bins
    kc = np.round(f_cq * (n_fft / sr)).astype(int)
    c_mat = np.zeros([n_bins, int(np.round(n_fft / 2))])
    for k in range(1, kc.shape[0] - 1):
        l1 = kc[k] - kc[k - 1]
        w1 = scipy.signal.triang((l1 * 2) + 1)
        l2 = kc[k + 1] - kc[k]
        w2 = scipy.signal.triang((l2 * 2) + 1)
        wk = np.hstack(
            [w1[0:l1], w2[l2:]])  # concatenate two halves. l1 and l2 are different because of the log-spacing
        if (kc[k + 1] + 1) > c_mat.shape[1]: # if out of matrix shape, continue
            continue
        c_mat[k - 1, kc[k - 1]:(kc[k + 1] + 1)] = wk / np.sum(wk)  # normalized to unit sum;
    return c_mat, f_cq  # matrix with triangular filterbank

def logf_stft(x, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr):
    """
    Logf-stft
    Based on https://github.com/mcartwright/dafx2018_adt/blob/master/large_vocab_adt_dafx2018/features.py
    @param x: array
    @param n_fft: int
    @param win_length: int
    @param hop_length: int
    @param n_bins_per_octave: int
    @param n_octaves: int
    @param f_min: float
    @param sr: float. sample rate
    @return x_cq_spec: logf-stft
    """
    if not _HAS_LIBROSA:
        logger.warning("Librosa is not installed. Please install it to use the logf-stft feature.")
        return None

    f_win = scipy.signal.hann(win_length)
    x_spec = librosa.stft(x,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=f_win)
    x_spec = np.abs(x_spec) / (2 * np.sum(f_win))

    # multiply stft by constant-q filterbank
    f_cq_mat, f_cq = cq_matrix(n_bins_per_octave, n_octaves * n_bins_per_octave, f_min, n_fft, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])
    stft = librosa.power_to_db(x_cq_spec).astype('float32')

    return stft, f_cq

def onset_strength_spec(x, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr, mean_filter_size):
    """
    Onset strength spectrogram
    Based on https://github.com/mcartwright/dafx2018_adt/blob/master/large_vocab_adt_dafx2018/features.py
    @param x: array
    @param n_fft: int
    @param win_length: int
    @param hop_length: int
    @param n_bins_per_octave: int
    @param n_octaves: int
    @param f_min: float
    @param sr: float. sample rate
    @param mean_filter_size: int. dt in the differential calculation
    @return od_fun: multi-band onset strength spectrogram
    @return f_cq: frequency bins of od_fun
    """
    if not _HAS_LIBROSA:
        logger.warning("Librosa is not installed. Please install it to use the logf-stft feature.")
        return None

    f_win = scipy.signal.hann(win_length)
    x_spec = librosa.stft(x,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=f_win)
    x_spec = np.abs(x_spec) / (2 * np.sum(f_win))

    # multiply stft by constant-q filterbank
    f_cq_mat, f_cq = cq_matrix(n_bins_per_octave, n_octaves * n_bins_per_octave, f_min, n_fft, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])

    # subtract moving mean: difference between the current frame and the average of the previous mean_filter_size frames
    b = np.concatenate([[1], np.ones(mean_filter_size, dtype=float) / -mean_filter_size])
    od_fun = scipy.signal.lfilter(b, 1, x_cq_spec, axis=1)

    # half-wave rectify
    od_fun = np.maximum(0, od_fun)

    # post-process OPs
    od_fun = np.log10(1 + 1000 * od_fun)  ## log scaling
    od_fun = np.abs(od_fun).astype('float32')
    od_fun = np.moveaxis(od_fun, 1, 0)
    # clip
    # FIXME check value of 2.25
    od_fun = np.clip(od_fun / 2.25, 0, 1)

    return od_fun, f_cq

def reduce_f_bands_in_spec(freq_out, freq_in, S):
    """
    @param freq_out:        band center frequencies in output spectrogram
    @param freq_in:         band center frequencies in input spectrogram
    @param S:               spectrogram to reduce
    @returns S_out:         spectrogram reduced in frequency
    """

    if len(freq_out) >= len(freq_in):
        warnings.warn(
            "Number of bands in reduced spectrogram should be smaller than initial number of bands in spectrogram")

    n_timeframes = S.shape[0]
    n_bands = len(freq_out)

    # find index of closest input frequency
    freq_out_idx = np.array([], dtype=int)

    for f in freq_out:
        freq_out_idx = np.append(freq_out_idx, np.abs(freq_in - f).argmin())

    # band limits (not center)
    freq_out_band_idx = np.array([0], dtype=int)

    for i in range(len(freq_out_idx) - 1):
        li = np.ceil((freq_out_idx[i + 1] - freq_out_idx[i]) / 2) + freq_out_idx[i]  # find left border of band
        freq_out_band_idx = np.append(freq_out_band_idx, [li])

    freq_out_band_idx = np.append(freq_out_band_idx, len(freq_in))  # add last frequency in input spectrogram
    freq_out_band_idx = np.array(freq_out_band_idx, dtype=int)  # convert to int

    # init empty spectrogram
    S_out = np.zeros([n_timeframes, n_bands])

    # reduce spectrogram
    for i in range(len(freq_out_band_idx) - 1):
        li = freq_out_band_idx[i] + 1  # band left index
        if i == 0: li = 0
        ri = freq_out_band_idx[i + 1]  # band right index
        if li >= ri: # bands out of range
            S_out[:,i] = 0
        else:
            S_out[:, i] = np.max(S[:, li:ri], axis=1)  # pooling

    return S_out

def detect_onset(onset_strength):
    """
    Detects onset from onset strength envelope

    """
    if not _HAS_LIBROSA:
        logger.warning("Librosa is not installed. Please install it to use the logf-stft feature.")
        return None

    n_timeframes = onset_strength.shape[0]
    n_bands = onset_strength.shape[1]

    onset_detect = np.zeros([n_timeframes, n_bands])

    for band in range(n_bands):
        time_frame_idx = librosa.onset.onset_detect(onset_envelope=onset_strength.T[band, :])
        onset_detect[time_frame_idx, band] = 1

    return onset_detect

def map_onsets_to_grid(grid, onset_strength, onset_detect, hop_length, n_fft, sr):
    """
    Maps matrices of onset strength and onset detection into a grid with a lower temporal resolution.
    @param grid:                 Array with timestamps
    @param onset_strength:       Matrix of onset strength values (n_timeframes x n_bands)
    @param onset_detect:         Matrix of onset detection (1,0) (n_timeframes x n_bands)
    @param hop_length:
    @param n_fft
    @return onsets_grid:         Onsets with respect to lines in grid (len_grid x n_bands)
    @return intensity_grid:      Strength values for each detected onset (len_grid x n_bands)
    """
    if not _HAS_LIBROSA:
        logger.warning("Librosa is not installed. Please install it to use the logf-stft feature.")
        return None

    if onset_strength.shape != onset_detect.shape:
        warnings.warn(
            f"onset_strength shape and onset_detect shape must be equal. Instead, got {onset_strength.shape} and {onset_detect.shape}")

    n_bands = onset_strength.shape[1]
    n_timeframes = onset_detect.shape[0]
    n_timesteps = len(grid) - 1 # last grid line is first line of next bar

    # init intensity and onsets grid
    strength_grid = np.zeros([n_timesteps, n_bands])
    onsets_grid = np.zeros([n_timesteps, n_bands])

    # time array
    time = librosa.frames_to_time(np.arange(n_timeframes), sr=sr,
                                  hop_length=hop_length, n_fft=n_fft)

    #FIXME already defined in io_helpers. cannot be imported here because io_helpers has a HVO_Sequence() import
    def get_grid_position_and_utiming_in_hvo(start_time, grid):
        """
        Finds closes grid line and the utiming deviation from the grid for a queried onset time in sec

        @param start_time:                  Starting position of a note
        @param grid:                        Grid lines (list of time stamps in sec)
        @return tuple of grid_index,        the index of the grid line closes to note
                and utiming:                utiming ratio in (-0.5, 0.5) range
        """
        grid_index, grid_sec = find_nearest(grid, start_time)

        utiming = start_time - grid_sec  # utiming in sec

        if utiming < 0:  # Convert to a ratio between (-0.5, 0.5)
            if grid_index == 0:
                utiming = 0
            else:
                utiming = utiming / (grid[grid_index] - grid[grid_index - 1])
        else:
            if grid_index == (grid.shape[0] - 1):
                utiming = utiming / (grid[grid_index] - grid[grid_index - 1])
            else:
                utiming = utiming / (grid[grid_index + 1] - grid[grid_index])

        return grid_index, utiming

    # map onsets and strength into grid
    for band in range(n_bands):
        for timeframe_idx in range(n_timeframes):
            if onset_detect[timeframe_idx, band]:  # if there is an onset detected, get grid index and utiming
                grid_idx, utiming = get_grid_position_and_utiming_in_hvo(time[timeframe_idx], grid)
                if grid_idx == n_timesteps : continue # in case that a hit is assigned to last grid line
                strength_grid[grid_idx, band] = onset_strength[timeframe_idx, band]
                onsets_grid[grid_idx, band] = utiming

    return strength_grid, onsets_grid

def get_hvo_idxs_for_voice(voice_idx, n_voices):
    """
    Gets index for hits, velocity and offsets for a voice. Used for copying hvo values from a voice from an
    hvo_sequence to another one.
    """
    h_idx = voice_idx
    v_idx = [_ + n_voices for _ in voice_idx]
    o_idx = [_ + 2 * n_voices for _ in voice_idx]

    return h_idx, v_idx, o_idx


########################################################################################################################
# Argand Tools for Balance, Evenness and Entropy
########################################################################################################################
from bokeh.palettes import Category10
from bokeh.models import Panel, Range1d
from bokeh.plotting import figure


#%%
def get16stepSegs(hvo_seq_): # ported to hvo_seq
    nsteps = hvo_seq_.number_of_steps
    # full multiple of 16
    if nsteps % 16 != 0:
        hvo_seq_.adjust_length(16 * (nsteps // 16 + 1))
        nsteps = hvo_seq_.number_of_steps

    _1bar_hvo_seqs = []
    for start in range(0, nsteps, 16):
        temp = hvo_seq_.copy()
        temp.hvo = temp.hvo[start:start+16, :]
        _1bar_hvo_seqs.append(temp)
    return _1bar_hvo_seqs



def convert_1d_pattern_to_argand_vector(pattern, offsets=None):
    assert pattern.ndim == 1, "pattern must be 1d array"
    assert pattern.shape[0] == 16, "pattern must be 16 elements long"
    assert offsets is None or offsets.shape[0] == 16, "offsets must be 16 elements long"

    hit_indices = np.where(pattern == 1)
    if offsets is not None:
        timings = np.array([index + offsets[index] for index in hit_indices])
    else:
        timings = np.array([index for index in hit_indices])

    return np.exp(2 * np.pi * 1j * timings / pattern.shape[0])[0], hit_indices[0]

def getEntropyOf1dArgandVector(hit_locations):
    n_onsets = hit_locations.shape[0]
    IOIs = sorted((hit_locations[1:] - hit_locations[:-1]))
    unique_IOIs, counts = np.unique(IOIs, return_counts=True)
    histogram = np.zeros((16))
    histogram[unique_IOIs] = counts
    normalized_histogram = histogram / n_onsets
    H_x = - np.log(normalized_histogram) * normalized_histogram /np.log(16)
    return np.nansum(H_x)

def getBalanceEvennessOf1dArgandVector(_1d_argand_vector):
    dft = np.fft.fft(_1d_argand_vector)
    K = _1d_argand_vector.shape[0]
    balance = 1 - np.abs(dft[0]) / K
    evenness = np.abs(dft[1]) / K
    return balance, evenness


def getBalanceEvennessEntropyOfHVO_Seq(_1_bar_hvo_seq, quantize_offsets=False, tappify=False, need_plot_data=False,
                                       shift_colors_by=0,
                                       identifier="", marker_size=10, marker_type="circle",
                                       use_primary_colors=True, radius=1):
    """If more than 1 voice in hvo_seq and tappify is false,
    then the argand vector will be the appandation of the argand vectors of each voice

Args:
        hvo_seq (HVO_Seq): [description]
        tappify (bool, optional): [description]. Defaults to False.

Returns:
        tuple: (balance list, evenness list, entropy list) where each index corresponds to a 1bar segment
    """

    _1_bar_hvo_seq.adjust_length(16)
    num_voices = _1_bar_hvo_seq.number_of_voices
    plot_data = []
    if use_primary_colors:
        colors = Category10[10]
    else:
        colors = [
          '#8B0000', # Dark red (high contrast),
          '#FFC0CB', # Pink (low contrast)
          '#FFD700', # Gold (high contrast)
          '#87CEEB', # Sky blue (low contrast)
          '#FF8C00', # Dark orange (high contrast)
          '#6A5ACD', # Slate blue (low contrast)
          '#00CED1', # Dark turquoise (high contrast)
          '#DB7093', # Pale violet red (low contrast)
          '#FF69B4', # Hot pink (high contrast)
          '#00FF7F', # Spring green (low contrast)
          '#9932CC', # Dark orchid (high contrast)
          '#00BFFF', # Deep sky blue (low contrast)
          '#FF1493', # Deep pink (high contrast)
          '#ADFF2F', # Green yellow (low contrast)
          '#FF00FF'  # Magenta (low contrast)
         ]

    colors = colors[shift_colors_by:]

    drum_voices = "Mixed " + " & ".join([key for key in _1_bar_hvo_seq.drum_mapping.keys()])
    # remove ints from drum_voices
    drum_voices = ''.join([i for i in drum_voices if not i.isdigit()])
    # Every word should be capitalized only at the beginning of each word
    drum_voices = drum_voices.title()

    if tappify:
        tapped = _1_bar_hvo_seq.flatten_voices(reduce_dim=True)
        hits = tapped[:, 0]
        offsets = tapped[:, 2]
        if quantize_offsets:
            offsets *= 0

        hit_vels = tapped[:, 1]
        pattern_argand_vec, hit_indices = convert_1d_pattern_to_argand_vector(hits, offsets)

        if need_plot_data:
            for hit_indices_ix, hit_index in enumerate(hit_indices):
                plot_data.append(
                    {
                        "argand": pattern_argand_vec[hit_indices_ix],
                        "hit_index": hit_index,
                        "hit_vel": hit_vels[hit_indices_ix],
                        "color": colors[0],
                        "label": drum_voices,
                        "identifier": identifier,
                        "marker_type": marker_type,
                        "marker_size": marker_size,
                        "radius": radius
                    }
                )

    else:
        num_voices = _1_bar_hvo_seq.number_of_voices
        drum_mapping = list(_1_bar_hvo_seq.drum_mapping.keys())
        hit_indices, hit_vels = np.array([]), np.array([])
        pattern_argand_vec_ = np.zeros((16))*1j
        for voice_ix in range(num_voices):
            hits = _1_bar_hvo_seq.get("h")[:, voice_ix]
            if np.sum(hits) > 0:
                vels = _1_bar_hvo_seq.get("v")[:, voice_ix]
                offsets = _1_bar_hvo_seq.get("o")[:, voice_ix]
                if quantize_offsets:
                    offsets *= 0
                voice_pattern_argand_vec, voice_hit_indices = convert_1d_pattern_to_argand_vector(hits, offsets)
                pattern_argand_vec_[voice_hit_indices] += voice_pattern_argand_vec
                hit_indices = np.append(hit_indices, voice_hit_indices.astype(int))
                hit_vels = np.append(hit_vels, vels[voice_hit_indices])
                if need_plot_data:
                    for hit_indices_ix, hit_index in enumerate(voice_hit_indices):
                        plot_data.append(
                            {
                                "argand": voice_pattern_argand_vec[hit_indices_ix],
                                "hit_index": hit_index,
                                "hit_vel": vels[hit_index],
                                "color": colors[voice_ix],
                                "label": drum_mapping[voice_ix],
                                "identifier": identifier,
                                "marker_type": marker_type,
                                "marker_size": marker_size,
                                "radius": radius
                            }
                        )

        pattern_argand_vec = np.array([pattern_argand_vec_[ix] for ix in range(16) if np.abs(pattern_argand_vec_[ix]) != 0])

    balance, evenness = getBalanceEvennessOf1dArgandVector(pattern_argand_vec)
    entropy = getEntropyOf1dArgandVector(hit_indices.astype(int))

    if need_plot_data:
        return np.round(balance, 2), np.round(evenness, 2), np.round(entropy, 2), plot_data
    else:
        return np.round(balance, 2), np.round(evenness, 2), np.round(entropy, 2), None

def draw1BarArgandVectorPlot(plot_data_, title="", plot_width=400, plot_height=400, line_width=0.3, line_color="black",
                             color_with_vel_values=False):

    if title == "":
        title = "--"
    fig = figure(plot_width=plot_width, plot_height=plot_height, title=title)

    find_unique_radii = set([plot_datum["radius"] for plot_datum in plot_data_])
    max_radius = max(find_unique_radii)

    for radius in find_unique_radii:
        if radius == max_radius:
            line_width_ = line_width
        else:
            line_width_ = line_width / 2
        # draw unit circle
        fig.line(np.cos(np.linspace(0, 2*np.pi, 100)) * radius,
                 np.sin(np.linspace(0, 2*np.pi, 100)) * radius,
                 color=line_color, line_width=line_width_)

        # draw markers on the unit circle separated equally by 22.5 degrees
        xs = np.cos(np.linspace(0, 2*np.pi, 17)) * radius
        ys = np.sin(np.linspace(0, 2*np.pi, 17)) * radius
        fig.scatter(x=xs, y=ys, radius=0.005, color=line_color)

    # draw circle at argand_val (polar coordinates)
    for plot_datum in plot_data_:
        argand_val = plot_datum["argand"]
        legend_label = plot_datum["label"]
        s = fig.scatter(x=[argand_val.real * plot_datum["radius"]], y=[argand_val.imag * plot_datum["radius"]],
                        color=plot_datum["color"],
                    fill_alpha=plot_datum["hit_vel"] if color_with_vel_values else 1,
                        line_width=line_width,
                        legend=f"{plot_datum['label']} ({plot_datum['identifier']})",
                        marker=plot_datum["marker_type"], size=plot_datum["marker_size"])

    fig.legend.click_policy="hide"
    # legend with font size 8
    fig.legend.label_text_font_size = "8pt"
    fig.legend.glyph_height = 8
    fig.legend.glyph_width = 8
    fig.legend.spacing = 0
    fig.legend.padding = 0
    fig.legend.margin = 0

    # legend in a single line
    fig.legend.location = "top_left"

    # x and y axis must have same range
    fig.x_range = fig.y_range = Range1d(-1.5, 1.5)



    # get rid of ticks and grid
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.grid.visible = False

    # draw a vertical and horizontal line at x=0, y=0
    fig.line(x=[0, 0], y=[-max_radius, max_radius], color='black', line_width=0.3, line_dash=(2, 2))
    fig.line(x=[-max_radius, max_radius], y=[0, 0], color='black', line_width=0.3, line_dash=(2, 2))

    return fig

def argandAnalysisWithPlots(first_hvo_seq, first_identifier="(A)", second_hvo_seq=None, second_identifier="(B)",
                            quantize=False, quantize_reference=False, flatten=False, flatten_reference=False, plot_width=400, plot_height=400,
                            title="", marker_size=12, line_width=0.3,
                            line_color="black", color_with_vel_values=False,
                            show_balance_evenness_entropy=True, need_title=False):
    balances_a, evennesses_a, entropies_a = [], [], []
    balances_b, evennesses_b, entropies_b = [], [], []
    argand_tabs = []

    if second_hvo_seq is not None:
        ref_plots_per_bar = []
        for bar_hs in get16stepSegs(second_hvo_seq):
            ref_Bal, ref_Ev, refEnt, ref_plot_data = getBalanceEvennessEntropyOfHVO_Seq(
                bar_hs, quantize_offsets=quantize_reference, tappify=flatten_reference, need_plot_data=True,
                identifier=second_identifier,marker_type="circle_x", marker_size=marker_size, use_primary_colors=False
                , radius=0.6) #, shift_colors_by=len(hvo_seq.drum_mapping.keys()))
            balances_b.append(ref_Bal)
            evennesses_b.append(ref_Ev)
            entropies_b.append(refEnt)
            ref_plots_per_bar.append(ref_plot_data)

    for bar_ix, bar_hs in enumerate(get16stepSegs(first_hvo_seq)):
        balance, evenness, entropy, plot_data = getBalanceEvennessEntropyOfHVO_Seq(
            bar_hs, quantize_offsets=quantize, tappify=flatten, need_plot_data=True,
            identifier=first_identifier, marker_type="circle_x", marker_size=int(marker_size*1.2), use_primary_colors=True, radius=0.8)
        title_ = f"{title} \n"
        title_ += f"{first_identifier} \n"
        if show_balance_evenness_entropy:
            title_ += "\n Bar {bar_ix + 1}, Balance = {balance}, Evenness = {evenness}, Entropy = {entropy} \n"

        balances_a.append(balance)
        evennesses_a.append(evenness)
        entropies_a.append(entropy)
        if balances_b:
            if bar_ix < len(balances_b):
                ref_plot_data = ref_plots_per_bar[bar_ix]
                plot_data.extend(ref_plot_data)
                title_ += f"{second_identifier} "
                if show_balance_evenness_entropy:
                    title_ += "\n Bar {bar_ix + 1}, Balance = {balances_b[bar_ix]}, Evenness = {evennesses_b[bar_ix]}, Entropy = {entropies_b[bar_ix]} \n"
            else:
                title_ += f"No corresponding bar in {second_identifier} for bar {bar_ix+1} in {first_identifier}"

        if second_hvo_seq is not None:
            ref_plot_data = ref_plots_per_bar[bar_ix]
            plot_data.extend(ref_plot_data)
        argand_tabs.append(draw1BarArgandVectorPlot(
            plot_data, title=title_ if need_title else "",
            plot_width=plot_width, plot_height=plot_height, line_width=line_width, line_color=line_color,
            color_with_vel_values=color_with_vel_values))

    from bokeh.models import Panel, Tabs
    tabs = Tabs(tabs=[Panel(child=fig, title=f"Bar {bar_ix+1}") for bar_ix, fig in enumerate(argand_tabs)])
    return balances_a, evennesses_a, entropies_a, tabs


def argandAnalysis(hvo_seq, quantize=False, flatten=False):
    balances, evennesses, entropies = [], [], []
    argand_tabs = []


    for bar_ix, bar_hs in enumerate(get16stepSegs(hvo_seq)):
        balance, evenness, entropy, _ = getBalanceEvennessEntropyOfHVO_Seq(bar_hs, quantize_offsets=quantize, tappify=flatten, need_plot_data=False)
        balances.append(balance)
        evennesses.append(evenness)
        entropies.append(entropy)

    return balances, evennesses, entropies

def jaccard_similarity(a, b):
    set_a = set([i for i, x in enumerate(a) if x == 1])
    set_b = set([i for i, x in enumerate(b) if x == 1])
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
    return dp[m][n]

