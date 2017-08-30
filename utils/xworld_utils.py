import numpy as np
import torch
from torch.autograd import Variable

def compute_block_alignment(filter_sizes, strides, block_size):
    """
    Given the size of a block (grid) on the original input image,
    the sizes of filters that are used to convolve the image, and the strides
    of the corresponding convolutions, automatically compute the final
    filter size and the final stride so that each pixel on the final conv
    layer represents exactly the image feature within a block in the input image
    """
    final_filter_size = block_size
    final_stride = block_size
    for i in range(len(filter_sizes)):
        final_filter_size = (final_filter_size - filter_sizes[i]) // strides[i] + 1
        final_stride      = (final_stride - 1) // strides[i] + 1
    assert(final_filter_size > 0 and final_stride > 0)
    return final_filter_size, final_stride

def xwd_random_step(env):
    return env.take_actions({'action':np.random.randint(0, 4), 'pred_sentence':''})

def xwd_get_state(config, env, vocab, command):
    # raw state
    env_state = env.get_state()

    task = env_state['task']

    # ===== Empty frame without seeing a navigation command for the current episode =====
    # - NOTE: This is weird because we have to take random steps in oder to change 
    #         the current state. So, one has to define how to take a random step 
    #         for each task. Ideally, the robot should receive either a command or 
    #         a question in the very beginning of an episode.
    if not task and command is None:
        return None, None, None, None

    # ===== When the program reaches here, it means the robot has a task at this frame =====
    # Case 1: frame with a recognition task (and possibly also a continuing navigation task)
    if 'XWorldRec' in task:
        tokens = env_state['sentence'].lower().split()
        question = vocab.convert_to_idx(tokens[:-1], unk='#oov#').unsqueeze(1) # [seqlen x 1]
        answer = vocab.convert_to_idx(tokens[-1:], unk='#oov#')
    # Case 2: first frame of a navigation task
    elif 'XWorldNav' in task:
        assert command is None or env_state['sentence'] == 'Well done .', 'Each episode should only have a single navigation task'
        tokens = env_state['sentence'].lower().split()
        command = vocab.convert_to_idx(tokens, unk='#oov#').unsqueeze(1) # [seqlen x 1]
        question, answer = None, None
    # Case 3: continuing frame of a navigation task
    else:
        assert command is not None, 'Should not reach here with command being {}'.format(command)
        question, answer = None, None
    
    image = torch.Tensor(env_state['screen']).view(1, 3, config.img_size, config.img_size)

    return image, command, question, answer
