# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------
'''
meta
0 hold
1 stand
2 sit
3 ride
4 walk
5 look
6 hit -
7 eat -
8 jump
9 lay
10 talk_on_phone
11 carry
12 throw
13 catch
14 cut -
15 run
16 work_on_computer
17 ski
18 surf
19 skateboard
20 smile
21 drink
22 kick
23 point
24 reads
25 snowboard

hoi_list = 
{0: u'surf_instr', 1: u'ski_instr', 2: u'cut_instr', 3: u'walk', 4: u'cut_obj', 5: u'ride_instr', 6: u'talk_on_phone_instr', 7: u'kick_obj', 8: u'work_on_computer_instr', 9: u'eat_obj', 10: u'sit_instr', 11: u'jump_instr', 12: u'lay_instr', 13: u'drink_instr', 14: u'carry_obj', 15: u'throw_obj', 16: u'eat_instr', 17: u'smile', 18: u'look_obj', 19: u'hit_instr', 20: u'hit_obj', 21: u'snowboard_instr', 22: u'run', 23: u'point_instr', 24: u'read_obj', 25: u'hold_obj', 26: u'skateboard_instr', 27: u'stand', 28: u'catch_obj'}

'''
def apply_prior(obj, prediction):
    # obj class from 0-80, including _background_

    if obj != 32: # not a snowboard, then the action is impossible to be snowboard
        prediction[21] = 0

    if obj != 74: # not a book, then the action is impossible to be read
        prediction[24] = 0

    if obj != 33: # not a sports ball, then the action is impossible to be kick
        prediction[7] = 0   

    if (obj != 41) and (obj != 40) and (obj != 42) and (obj != 46): # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        prediction[13] = 0       

    if obj != 37: # not a skateboard, then the action is impossible to be skateboard
        prediction[26] = 0    

    if obj != 38: # not a surfboard, then the action is impossible to be surfboard
        prediction[0] = 0  
                            
    if obj != 31: # not a ski, then the action is impossible to be ski
        prediction[1] = 0      
                             
    if obj != 64: # not a laptop, then the action is impossible to be work on computer
        prediction[8] = 0
                        
    if (obj != 77) and (obj != 43) and (obj != 44): # not 'scissors', 'fork', 'knife', then the action is impossible to be cur instr
        prediction[2] = 0
                        
    if (obj != 33) and (obj != 30): # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        prediction[15] = 0
        prediction[28] = 0
                              
    if obj != 68: # not a cellphone, then the action is impossible to be talk_on_phone
        prediction[6] = 0   
                            
    if (obj != 14) and (obj != 61) and (obj != 62) and (obj != 60) and (obj != 58)  and (obj != 57): # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        prediction[12] = 0
                            
    if (obj != 32) and (obj != 31) and (obj != 37) and (obj != 38): # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        prediction[11] = 0   
   
    if (obj != 47) and (obj != 48) and (obj != 49) and (obj != 50) and (obj != 51) and (obj != 52) and (obj != 53) and (obj != 54) and (obj != 55) and (obj != 56): # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        prediction[9] = 0 

    if (obj != 43) and (obj != 44) and (obj != 45): # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        prediction[16] = 0 
            
    if (obj != 39) and (obj != 35): # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        prediction[19] = 0 

    if (obj != 33): # not 'sports ball, then the action is impossible to be hit_obj
        prediction[20] = 0 
                            
                            
    if (obj != 2) and (obj != 4) and (obj != 6) and (obj != 8) and (obj != 9) and (obj != 7) and (obj != 5) and (obj != 3) and (obj != 18) and (obj != 21): # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        prediction[5] = 0 
                            
    if (obj != 2) and (obj != 4) and (obj != 18) and (obj != 21) and (obj != 14) and (obj != 57) and (obj != 58) and (obj != 60) and (obj != 62) and (obj != 61) and (obj != 29) and (obj != 27) and (obj != 25): # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        prediction[10] = 0 
        
    if (obj == 1): # cut_object not on 'person'
        prediction[4] = 0 
    
    return prediction
                            


