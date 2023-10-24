from env import play_wvtr,play_random

paras = {}

# ----- Default Parameters -----
paras['action_space'] = [0,1]
paras['init_state'] = 0
paras['dim'] = 3

paras['lam'] = 0.001
paras['beta'] = 1
paras['sigmamin'] = 0.01
paras['gamma'] = 0.5
paras['M'] = 4
# ----- Default Parameters -----

# ----- Single Play ------------
# --- Spicify the parameters for environment and algorithms here
paras['S'] = 10
paras['state_space'] = range(paras['S'])
paras['H'] = 100
paras['K'] = 1000
paras['num_path'] = 10
# --- Spicify the parameters for environment and algorithms here

# UCRL-WVTR
paras.update({'M':4, 'gamma':0.5})
play_wvtr(paras)
# UCRL-WVTR without HOME
paras.update({'M':2, 'gamma':0.5})
play_wvtr(paras)
# UCRL-VTR
paras.update({'M':1, 'gamma':0})
play_wvtr(paras)
# Random
play_random(paras)