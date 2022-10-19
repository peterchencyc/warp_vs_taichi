import warp as wp
import warp.torch
from timer_cm import Timer
import taichi as ti
import numpy as np
import argparse
import h5py

parser = argparse.ArgumentParser(description = "warp_vs_taichi")
parser.add_argument("-lang", type = str, choices=['warp','taichi'], required=True)
args = parser.parse_args()

h5file = h5py.File("./double_cone_highRes_initial_sampling.h5", 'r')
particle_q_np = h5file['q']
particle_q_np = np.transpose(particle_q_np) # the original 3 by n shape is unfriendly for warp; transpose it to make it n by 3

if args.lang == 'warp':
    device = "cuda"
    wp.init()

    particle_q = wp.from_numpy(particle_q_np, dtype=wp.vec3, device=device)

    timer = Timer("warm start")
    num_wm = 1001
    for i in range(num_wm):
        with timer.child('warm start cost'):
            particle_q_t = wp.to_torch(particle_q)
            particle_q = wp.from_torch(particle_q_t)
    for i in range(num_wm):
        with timer.child('actual cost 1st try'):
            particle_q_t = wp.to_torch(particle_q)
            particle_q = wp.from_torch(particle_q_t)
    for i in range(num_wm):
        with timer.child('actual cost 2nd try'):
            particle_q_t = wp.to_torch(particle_q)
            particle_q = wp.from_torch(particle_q_t)
    timer.print_results()    
elif args.lang == 'taichi':
    ti.init(arch = ti.gpu)

    particle_q = ti.Vector.field(3, dtype=float, shape=(particle_q_np.shape[0])) # position
    particle_q.from_numpy(particle_q_np)

    timer = Timer("warm start")
    num_wm = 1001
    for i in range(num_wm):
        with timer.child('warm start cost'):
            particle_q_t = particle_q.to_torch()
            particle_q.from_torch(particle_q_t)
    for i in range(num_wm):
        with timer.child('actual cost 1st try'):
            particle_q_t = particle_q.to_torch()
            particle_q.from_torch(particle_q_t)
    for i in range(num_wm):
        with timer.child('actual cost 2nd try'):
            particle_q_t = particle_q.to_torch()
            particle_q.from_torch(particle_q_t)
    timer.print_results()    