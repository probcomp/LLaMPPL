import copy
import numpy as np
from .util import logsumexp

def smc(model, n_particles, ess_threshold=0.5):
    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    weights = [0.0 for _ in range(n_particles)]

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Step each particle
        for i, p in enumerate(particles):
            if not p.done_stepping():
                weights[i] += p.step()

        # Normalize weights
        W = np.array(weights)
        w_sum = logsumexp(W)
        normalized_weights = W - w_sum

        # Print the particles
        for i, p in enumerate(particles):
            print(f"Particle {i}: {p} (weight {weights[i]})")
        
        # print(model.llms[0].ctx)
        # Resample if necessary
        if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(n_particles):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(normalized_weights)
            particles = [copy.deepcopy(particles[np.random.choice(range(len(particles)), p=probs)]) for _ in range(n_particles)]
            weights = [w_sum - np.log(n_particles) for _ in range(n_particles)]

        

    # Return the particle with the highest weight
    return particles, weights
