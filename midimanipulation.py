# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:21:11 2019

@author: Garnet
"""
import mido
import numpy as np

INPUT_SIZE = 10
ACCEPTED_MESSAGES = ['end_of_track', 'program_change', 'control_change', 'note_on']

def merge_tracks(track1, track2):
    merge_time = t1_time = t2_time = 0    #time in seconds into each track
    i = j = 0    #index in track1 and track2 respectively
    result = mido.MidiTrack()
    name = mido.MetaMessage('track_name', 
                            name = track1.name + " and " + track2.name)
    result.append(name)
    # Put messages in order until we exhaust one track
    while i < len(track1) and j < len(track2):
        #Skip metamessages
        if track1[i].is_meta:
            i += 1
        elif track2[j].is_meta:
            j += 1
        else:
            if track1[i].time + t1_time <= track2[j].time + t2_time:
                time_dif = t1_time + track1[i].time - merge_time
                result.append(track1[i])
                t1_time += track1[i].time
                i += 1
                
            else:
                time_dif = t2_time + track2[j].time - merge_time
                result.append(track2[j])
                t2_time += track2[j].time
                j += 1
                
            result[-1].time = time_dif
            merge_time += time_dif
            
    # Append the rest of the un-exhausted track
    # We only need to fix the time value for the first note
    # The rest already have the correct time relative to the previous note
    if i < len(track1):
        time_dif = t1_time + track1[i].time - merge_time
        result.append(track1[i])
        result[-1].time = time_dif
        for x in range(i+1, len(track1)):
            if not track1[x].is_meta:
                result.append(track1[x])
    else:
        time_dif = t2_time + track2[j].time - merge_time
        result.append(track2[j])
        result[-1].time = time_dif
        for x in range(j+1, len(track2)):
            if not track2[x].is_meta:   
                result.append(track2[x])
    return result


def get_merged_piano_tracks(mid):
    piano_tracks = []
    for track in mid.tracks:
        if len(track) > 10 and 'piano' in track.name.lower():
            piano_tracks.append(track)
    if len(piano_tracks) == 0:
        return None
    else:
        while len(piano_tracks) > 1:
            piano_tracks[0] = merge_tracks(piano_tracks[0],
                                            piano_tracks.pop(-1))
        return piano_tracks[0]
    
    
def tensor_to_msg_list(tensor):
    """ Returns a mido.Message() if tensor.shape[0] == 1.
        Otherwise, returns a mido.MidiTrack()
    Transforms the given tensor into the appropriate mido class
    Requires the tensor to be cleaned by clean_tensor() first
    mido will throw an error if the values are not in an appropriate range
    """
    if tensor.shape == (INPUT_SIZE, ):
        msg_type = ACCEPTED_MESSAGES[tensor[:4].argmax()]
        msg_time = tensor[4]
        if msg_type == "end_of_track":
            msg = mido.MetaMessage("end_of_track")
        else:
            msg = mido.Message(msg_type, time=msg_time)
            if msg_type == 'program_change':
                msg.program = int(tensor[5])
            elif msg_type == 'control_change':
                msg.control = int(tensor[6])
                msg.value = int(tensor[7])
            elif msg_type == 'note_on':
                msg.note = int(tensor[8])
                msg.velocity = int(tensor[9])
        return msg
            
    else:
        track = []
        for i in tensor:
            track.append(tensor_to_msg_list(i))
        return track
    
    
def clean_tensor(tensor, track_time):
    """
    Exactly one of tensor[n][:4] should be treated as 1, the rest should be 0
    (take the largest value and set it to 1)
    tensor[4] should be in [0,1]
    Unless specified, tensor[5:] should be 0
    If tensor[1] == 1, tensor[5] should be in [0,127]
    if tensor[2] == 1, tensor[6] should be in [0, 127]
    If tensor[3] == 1, tensor[7:] should be in [0, 127]
    """
    result = np.zeros([1,INPUT_SIZE],dtype=int)
    if tensor.shape == (INPUT_SIZE, ):
        # Clamp the time value
        tensor[4] = max(0, min(1, tensor[4]))
        result[0][4] = round(tensor[4] * track_time)
        # A midi message can only have one type
        msg_type = tensor[:4].argmax()
        for i in range(4):
            if i == msg_type:
                result[0][i] = 1
            else:
                result[0][i] = 0
        # End of Song msg == 0
        if msg_type == 0:
            result[0][5:] = 0
        # Program msg === 1
        if msg_type == 1:
            tensor[5] = max(0, min(1, tensor[5]))
            result[0][5] = round(tensor[5] * 127)
        # Control msg === 2
        if msg_type == 2:
            tensor[6] = max(0, min(1, tensor[6]))
            tensor[7] = max(0, min(1, tensor[7]))
            result[0][6] = round(tensor[6] * 127)
            result[0][7] = round(tensor[7] * 127)
        # Note msg === 3
        if msg_type == 3:
            tensor[8] = max(0, min(1, tensor[8]))
            tensor[9] = max(0, min(1, tensor[9]))
            result[0][8] = round(tensor[8] * 127)
            result[0][9] = round(tensor[9] * 127)
        return result
    
    else:
        for i in tensor:
            x = clean_tensor(i, track_time)
            x = np.reshape(x, [1, INPUT_SIZE])
            result = np.concatenate((result, x), axis=0)
        result = np.delete(result, 0, 0)
        return result

def track_to_tensor(track):
    """
    Converts a mido midi track into a tensor for use in the
    neural network.
    """
    
    #Find how large our tensor needs to be
    messages = 0
    total_time = 0
    for msg in track:
        if msg.type in ACCEPTED_MESSAGES:
            messages += 1
            total_time += msg.time
    messages += 1 # for 'end_of_track' message

    # tensor[0:4] = msg.type, tensor[4] = % max_time,
    # tensor[5:8] = value for msg.type, tensor[8] = velocity
    # values for msg.type: program_change -? program, note_on -> note,
    #                       control_change -> control
    tensor = np.zeros((messages, INPUT_SIZE))
    i = 0
    for msg in track:
        for j in range(len(ACCEPTED_MESSAGES)):
            if msg.type == ACCEPTED_MESSAGES[j]:
                # Set message type input
                tensor[i][j] = 1
                # Set message time
                tensor[i][4] = msg.time/total_time
                # Set value
                if msg.type == 'program_change':
                    tensor[i][5] = msg.program / 127
                if msg.type == 'control_change':
                    tensor[i][6] = msg.control / 127
                    tensor[i][7] = msg.value/ 127
                if msg.type == 'note_on':
                    tensor[i][8] = msg.note / 127
                    tensor[i][9] = msg.velocity / 127
                i += 1
    # Add the end_of_track
    tensor[i][0] = 1
    return tensor  
    
    
def tensor_to_midi(tensor, track_time, name=None):
    tensor = clean_tensor(tensor, track_time)
    track = tensor_to_msg_list(tensor)
    mid = mido.MidiFile()
    mid.add_track()
    mid.tracks[0].append(mido.MetaMessage('track_name', name="testtrack"))
    for msg in track:
        mid.tracks[0].append(msg)
    if name != None:
        mid.save("{}.mid".format(name))
    return mid
    

if __name__ == "__main__":
    mid = mido.MidiFile("./chopin/chp_op18.mid")
    track = get_merged_piano_tracks(mid)
    total_time = 0
    fd = open("chp_op18_msgs.txt", "w")
    for msg in track:
        fd.write(str(msg))
        fd.write("\n")
        total_time += msg.time
    fd.close()
    tensor = track_to_tensor(track)
    conv_mid = tensor_to_midi(tensor, total_time, 'test')
    
    
    
    
    
    
    
    
    