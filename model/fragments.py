import numpy as np
import math


def calculate_fragments(sequence_len, num_fragments):
    '''
    The sequence must be divided into "num_fragments" fragments.
    Since seq_len/num won't be a perfect division, we take both
    floor and ceiling parts, in a way such that the sum of all
    fragments will be equal to the total sequence.
    '''

    fragment_size = sequence_len / num_fragments
    fragment_floor = math.floor(fragment_size)
    fragment_ceil = math.ceil(fragment_size)
    i_part, d_part = divmod(fragment_size, 1)

    frag_jump = np.zeros(num_fragments)

    upper = d_part * num_fragments
    upper = np.round(upper).astype(int)
    lower = (1 - d_part) * num_fragments
    lower = np.round(lower).astype(int)

    for i in range(lower):
        frag_jump[i] = fragment_floor
    for i in range(upper):
        frag_jump[lower + i] = fragment_ceil

    # Roll the scores, so that the larger fragments fall at
    # the center of the sequence. Should not make a difference.
    frag_jump = np.roll(frag_jump, -int(num_fragments * (1 - d_part) / 2))

    if frag_jump[num_fragments - 1] == 1:
        frag_jump[int(num_fragments / 2)] = 1

    return frag_jump.astype(int)


def compute_fragments(seq_len, action_state_size):
    # "action_fragments" contains the starting and ending frame of each action fragment
    frag_jump = calculate_fragments(seq_len, action_state_size)
    action_fragments = np.zeros((action_state_size, 2), dtype=int)
    frag_end, frag_next_begin = 0, 0
    max_frag_i = seq_len - 1
    max_i = action_fragments.shape[0] - 1
    for i, jump in enumerate(frag_jump):
        frag_next_begin += jump
        frag_end = frag_next_begin - int(0 < jump)

        frag_end = min(frag_end, max_frag_i)
        action_fragments[i, 1] = frag_end
        if i == max_i:
            break
        frag_next_begin = min(frag_next_begin, max_frag_i)
        action_fragments[i + 1, 0] = frag_next_begin

    return action_fragments


if __name__ == "__main__":
    pass
