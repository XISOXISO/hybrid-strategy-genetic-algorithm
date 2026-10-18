import math
import numpy as np

def _cpp_round_positive(value):
    return int(math.floor(value + 0.5))

class DifferentialEvolution:

    def __init__(self, bounds, objective_function, minimize=True, random_state=0, verbose=False, budget=1000, variant='LSHADE'):
        if not minimize or variant not in ('LSHADE', 'jSO'):
            raise ValueError('Requires minimization and LSHADE or jSO')
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or (not np.all(np.isfinite(self.bounds))) or (not np.all(self.bounds[:, 0] < self.bounds[:, 1])):
            raise ValueError('bounds must contain finite increasing pairs')
        self.objective = objective_function
        self.rng = np.random.default_rng(random_state)
        self.budget = int(budget)
        self.variant = variant
        self.history = {'best_values': [], 'best_params': []}

    def optimize(self):
        rng, bounds, limit = (self.rng, self.bounds, self.budget)
        dim = len(bounds)
        lower, upper = (bounds[:, 0], bounds[:, 1])
        nominal = _cpp_round_positive(18 * dim) if self.variant == 'LSHADE' else _cpp_round_positive(25 * math.log(dim) * math.sqrt(dim))
        if limit < nominal:
            raise ValueError(f"{self.variant}: budget {limit} is below the author's initial population {nominal}")
        population = lower + rng.random((nominal, dim)) * (upper - lower)
        fitness = np.asarray([float(self.objective(x)) for x in population])
        if not np.all(np.isfinite(fitness)):
            raise ValueError('DE objective returned a nonfinite value')
        used = nominal
        best_i = int(np.argmin(fitness))
        best_x, best_f = (population[best_i].copy(), float(fitness[best_i]))
        archive = []
        h = 6 if self.variant == 'LSHADE' else 5
        memory_f = np.full(h, 0.5 if self.variant == 'LSHADE' else 0.3)
        memory_cr = np.full(h, 0.5 if self.variant == 'LSHADE' else 0.8)
        memory_index = 0
        archive_ratio = 2.6 if self.variant == 'LSHADE' else 1.0
        archive_cap = _cpp_round_positive(archive_ratio * nominal)
        p_rate = 0.11 if self.variant == 'LSHADE' else 0.25
        p_size = _cpp_round_positive(nominal * p_rate)
        while used < limit:
            n = len(population)
            generation_fe = used
            order = np.argsort(fitness, kind='stable')
            trials, factors, crossovers = ([], [], [])
            for i in range(min(n, limit - used)):
                slot = int(rng.integers(h))
                mf, mcr = (float(memory_f[slot]), float(memory_cr[slot]))
                if self.variant == 'jSO' and slot == h - 1:
                    mf = mcr = 0.9
                cr = 0.0 if mcr < 0 else float(np.clip(rng.normal(mcr, 0.1), 0.0, 1.0))
                if self.variant == 'jSO':
                    if generation_fe < 0.25 * limit:
                        cr = max(cr, 0.7)
                    if generation_fe < 0.5 * limit:
                        cr = max(cr, 0.6)
                f = -1.0
                while f <= 0:
                    f = float(mf + 0.1 * rng.standard_cauchy())
                f = min(f, 1.0)
                if self.variant == 'jSO' and generation_fe < 0.6 * limit:
                    f = min(f, 0.7)
                candidates = order[:p_size]
                if self.variant == 'jSO' and generation_fe < 0.5 * limit:
                    candidates = candidates[candidates != i]
                pbest = int(rng.choice(candidates))
                r1 = int(rng.choice([j for j in range(n) if j != i]))
                pool = n + len(archive)
                r2 = int(rng.integers(pool))
                while r2 == i or r2 == r1:
                    r2 = int(rng.integers(pool))
                x2 = population[r2] if r2 < n else archive[r2 - n]
                fw = f
                if self.variant == 'jSO':
                    fw *= 0.7 if generation_fe < 0.2 * limit else 0.8 if generation_fe < 0.4 * limit else 1.2
                forced = int(rng.integers(dim))
                mask = rng.random(dim) < cr
                mask[forced] = True
                trial = np.where(mask, population[i] + fw * (population[pbest] - population[i]) + f * (population[r1] - x2), population[i])
                trial = np.where(trial < lower, (lower + population[i]) / 2, trial)
                trial = np.where(trial > upper, (upper + population[i]) / 2, trial)
                trials.append(trial)
                factors.append(f)
                crossovers.append(cr)
            values = []
            for trial in trials:
                value = float(self.objective(trial))
                if not math.isfinite(value):
                    raise ValueError('DE objective returned a nonfinite value')
                values.append(value)
                used += 1
                if value < best_f:
                    best_x, best_f = (trial.copy(), value)
            if len(trials) < n:
                break
            success_f, success_cr, improvements = ([], [], [])
            for i, (trial, value, f, cr) in enumerate(zip(trials, values, factors, crossovers)):
                if value < fitness[i]:
                    improvements.append(float(fitness[i] - value))
                    success_f.append(f)
                    success_cr.append(cr)
                    if archive_cap > 1:
                        if len(archive) < archive_cap:
                            archive.append(population[i].copy())
                        else:
                            archive[int(rng.integers(archive_cap))] = population[i].copy()
                if value <= fitness[i]:
                    population[i], fitness[i] = (trial, value)
            if improvements:
                weights = np.asarray(improvements) / sum(improvements)
                sf, sc = (np.asarray(success_f), np.asarray(success_cr))
                new_f = float(np.dot(weights, sf * sf) / np.dot(weights, sf))
                den_cr = float(np.dot(weights, sc))
                new_cr = -1.0 if den_cr == 0 else float(np.dot(weights, sc * sc) / den_cr)
                if self.variant == 'jSO':
                    new_f = (new_f + memory_f[memory_index]) / 2
                    new_cr = (new_cr + memory_cr[memory_index]) / 2
                memory_f[memory_index], memory_cr[memory_index] = (new_f, new_cr)
                memory_index = (memory_index + 1) % h
            target_n = max(4, _cpp_round_positive(nominal + (4 - nominal) * used / limit))
            if n > target_n:
                for _ in range(n - target_n):
                    worst = int(np.argmax(fitness))
                    population = np.delete(population, worst, axis=0)
                    fitness = np.delete(fitness, worst)
                archive_cap = int(len(population) * archive_ratio)
                del archive[archive_cap:]
                if self.variant == 'jSO':
                    p_rate = 0.25 * (1 - 0.5 * used / limit)
                p_size = max(2, _cpp_round_positive(len(population) * p_rate))
        self.history['best_values'].append(best_f)
        self.history['best_params'].append(best_x.copy())
        return (best_x, best_f)
