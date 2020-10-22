import numpy as np

'''
Generate an excitation schedule as firing rates for use as a TimedArray
Excitation is in the shape of cos(p)^cosexp for p=(0,pi)
Width is adjusted so that the half-cycle cosine occupies the desired fraction of the period (1/frequency)
Returns a T/dt x N_nodes matrix where each row is the rate of spiking across nodes for one time point
Amplitude is normalized to 1, so needs to be scaled outside this function
Schedule is generated nodes x time, and transposed on output

N_nodes:         number of output nodes
pattern:         'circular'  periodic with a phase gradient of 0-2pi across excited nodes
                 'random'    random permutation of time on output, 
dt:              time step in ms
T:               total time in ms
duty_cycle:      fraction of time occupied by excitation
frequency:       frequency of oscillation in Hz
ramp_on[0]:      fraction of excitation period doing a half sine ramp up of excitation density
ramp_off[0]:     fraction of excitation period doing a half cosine ramp down of excitation density
cosexp[1]:       Shape of excitation profile is cos^cosexp
phase_roll[2pi]: For circular pattern, phase across nodes varies by phase_roll 
                   0 for periodic excitation of all nodes in phase
                   2pi for simple circular advancement
                   >2pi for circular with a gap
                   2Npi for N simultaneous groups of excitation chasing each other around the ring
'''
def excitation_schedule(N_nodes,pattern,dt,T,duty_cycle,frequency,ramp_on=0,ramp_off=0,cosexp=1.,phase_roll=2.0*np.pi):

    nt = int(round(T/dt))
    t = np.linspace(0.0,0.001*T,nt,endpoint=False) # in s (T is in ms)
    nodeshift = np.linspace(0, phase_roll, num=N_nodes, endpoint=False)
    TIME, SHIFT = np.meshgrid(t, nodeshift)
    # Calculate phase to feed into cos()
    phase = 0.5 * ((np.pi + 2.0 * np.pi * frequency * TIME - SHIFT) % (2.0 * np.pi) - np.pi) / duty_cycle
    phase = np.clip(phase,-0.5*np.pi,0.5*np.pi)
    ex = np.cos(phase)
    
    if cosexp != 1.:
        ex = ex ** cosexp

    if pattern == 'random':
        ex = np.apply_along_axis(np.random.permutation, axis=1, arr=ex)
    
    if ramp_on or ramp_off:
        n_on = int(ramp_on * nt)
        n_off = int(ramp_off * nt)
        window = np.ones(nt)
        if n_on:
            window[:n_on] = np.sin(np.linspace(0,0.5*np.pi,n_on))
        if n_off:
            window[-n_off:] = np.cos(np.linspace(0,0.5*np.pi,n_off))
        window = np.tile(window,(N_nodes,1))
        ex *= window
        
    return ex.T