import numpy as np
import copy
from .util import logsumexp, softmax

def copy_with_index(particle, index):
    new_particle = copy.deepcopy(particle)
    new_particle.beam_idx = index
    new_particle.force_eos = False
    return new_particle

def beam(model, n_beam, n_explore=None, force_eos = False, mode = "beam"):
    # Create n_beam copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_beam)]

    if n_explore is None:
        n_explore = n_beam

    for (i, particle) in enumerate(particles):
        particle.mode = mode
        particle.start()
        particle.beam_idx = i+1
        particle.step()

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Create a super-list of particles that has n_beam copies of each
        super_particles = []
        for p in particles:
            p.beam_idx = 1
            super_particles.append(p)
            super_particles.extend([copy_with_index(p,i+2) for i in range(n_explore-1)])
            if force_eos:
                super_particles.append(copy.deepcopy(p))
                super_particles[-1].force_eos = True
        
        # Step each super-particle
        for i, p in enumerate(super_particles):
            # Step
            if not p.done_stepping():
                p.step()
            print(f"Particle {i}: {p} (weight {p.weight})")
        
        # Take the best n_beam particles
        super_particles.sort(key=lambda p: p.weight, reverse=True)
        particles = super_particles[:n_beam]

    # Return the particles
    return particles
