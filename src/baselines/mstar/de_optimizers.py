from __future__ import annotations
import math
import numpy as np
NAMES = ('JSO', 'LSHADE')

def round_positive(value):
    return int(math.floor(float(value) + 0.5))

def parameters(name, dimension=8, budget=500):
    if name not in NAMES or dimension < 1 or budget < 1:
        raise ValueError('Invalid DE name/dimension/budget')
    jso = name == 'JSO'
    nominal = max(4, round_positive(25 * math.log(dimension) * math.sqrt(dimension))) if jso else 18 * dimension
    return dict(algorithm=name, adaptation='discrete-HPO-v3', dimension=dimension, nominal_initial_population=nominal, initial_population=min(nominal, budget), minimum_population=4, memory_size=5, archive_rate=1.0 if jso else 1.4, initial_MF=0.3 if jso else 0.5, initial_MCR=0.8 if jso else 0.5, p_max=0.25 if jso else 0.11, p_min=0.125 if jso else 0.11, fixed_last_sampling_memory=0.9 if jso else None, boundary='parent-bound midpoint', population_schedule='linear in unique FE / MAXFE')

def p_rate(name, fraction):
    return 0.25 * (1 - 0.5 * fraction) if name == 'JSO' else 0.11

def jso_controls(f, cr, fraction):
    if fraction < 0.25:
        cr = max(cr, 0.7)
    elif fraction < 0.5:
        cr = max(cr, 0.6)
    if fraction < 0.6:
        f = min(f, 0.7)
    attraction = f * (0.7 if fraction < 0.2 else 0.8 if fraction < 0.4 else 1.2)
    return (f, cr, attraction)

def midpoint_bounds(mutant, parent, lower=0.0, upper=2.0):
    mutant = np.asarray(mutant, dtype=float)
    parent = np.asarray(parent, dtype=float)
    return np.where(mutant < lower, (parent + lower) / 2, np.where(mutant > upper, (parent + upper) / 2, mutant))

def mutation(parent, pbest, r1, r2, f, attraction):
    return parent + attraction * (pbest - parent) + f * (r1 - r2)

def lehmer(values, improvements):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(improvements, dtype=float)
    weights = weights / weights.sum()
    denominator = float(np.dot(weights, values))
    return float(np.dot(weights, values ** 2) / denominator) if denominator > 0 else None

class SuccessHistoryDE:

    def __init__(self, name, objective, seed, budget):
        self.name, self.objective, self.budget = (name, objective, int(budget))
        self.settings = parameters(name, 8, self.budget)
        self.rng = np.random.default_rng(seed)
        self.initial_n = self.settings['initial_population']
        self.mf = np.full(5, self.settings['initial_MF'], dtype=float)
        self.mcr = np.full(5, self.settings['initial_MCR'], dtype=float)
        self.memory_pos = 0
        self.archive = []
        self.generation = 0
        self.history = []
        self.best = float('inf')
        self.repaired = 0
        self.last_progress_fraction = 0.0
        self.current_p = self.settings['p_max']
        self.pop = np.empty((self.initial_n, 8))
        self.fitness = np.full(self.initial_n, float('inf'))

    def evaluate(self, raw):
        raw = np.clip(np.asarray(raw, dtype=float), 0.0, 2.0)
        genes, _, changed = self.objective.resolve(raw, True)
        score = float(self.objective(raw))
        self.repaired += int(changed)
        self.best = min(self.best, score)
        return (genes.astype(float), score)

    def sample_controls(self, fraction):
        index = int(self.rng.integers(5))
        if self.name == 'JSO' and index == 4:
            mu_f = mu_cr = 0.9
        else:
            mu_f, mu_cr = (self.mf[index], self.mcr[index])
        cr = 0.0 if mu_cr < 0 else float(np.clip(self.rng.normal(mu_cr, 0.1), 0.0, 1.0))
        while True:
            f = float(mu_f + 0.1 * self.rng.standard_cauchy())
            if math.isfinite(f) and f > 0:
                break
        f = min(f, 1.0)
        if self.name == 'JSO':
            return jso_controls(f, cr, fraction)
        return (f, cr, f)

    def update_memory(self, successful_f, successful_cr, improvements):
        if not successful_f:
            return
        index = self.memory_pos
        f = lehmer(successful_f, improvements)
        cr = lehmer(successful_cr, improvements)
        cr = -1.0 if cr is None else cr
        if self.name == 'JSO':
            self.mf[index] = (self.mf[index] + f) / 2
            self.mcr[index] = (self.mcr[index] + cr) / 2
        else:
            self.mf[index], self.mcr[index] = (f, cr)
        self.memory_pos = (index + 1) % 5

    def add_archive(self, parent, capacity):
        if capacity <= 1:
            return
        if len(self.archive) < capacity:
            self.archive.append(parent.copy())
        else:
            self.archive[int(self.rng.integers(capacity))] = parent.copy()

    def iter_evaluate(self, proposals):
        if hasattr(self.objective, 'prefetch'):
            self.objective.prefetch(proposals)
        for proposal in proposals:
            yield self.evaluate(proposal)

    def run(self):
        proposals = self.rng.uniform(0.0, 2.0, (self.initial_n, 8))
        for i, point in enumerate(self.iter_evaluate(proposals)):
            self.pop[i], self.fitness[i] = point
        while self.objective.fes < self.budget:
            n = len(self.pop)
            if n < 4:
                raise RuntimeError('DE population too small to evolve')
            self.generation += 1
            fraction = self.objective.fes / self.budget
            generation_p = self.current_p
            self.last_progress_fraction = fraction
            order = np.argsort(self.fitness, kind='stable')
            count = min(n, max(2, round_positive(n * self.current_p)))
            union = np.vstack([self.pop] + ([np.asarray(self.archive)] if self.archive else []))
            trials, fs, crs = ([], [], [])
            for i in range(n):
                f, cr, attraction = self.sample_controls(fraction)
                candidates = order[:count]
                if self.name == 'JSO' and fraction < 0.5:
                    candidates = candidates[candidates != i]
                pbest = int(candidates[int(self.rng.integers(len(candidates)))])
                r1 = int(self.rng.integers(n))
                while r1 == i:
                    r1 = int(self.rng.integers(n))
                r2 = int(self.rng.integers(len(union)))
                while r2 == i or r2 == r1:
                    r2 = int(self.rng.integers(len(union)))
                mutant = mutation(self.pop[i], self.pop[pbest], self.pop[r1], union[r2], f, attraction)
                mask = self.rng.random(8) < cr
                mask[int(self.rng.integers(8))] = True
                trial = midpoint_bounds(np.where(mask, mutant, self.pop[i]), self.pop[i])
                trials.append(trial)
                fs.append(f)
                crs.append(cr)
            capacity = round_positive(n * self.settings['archive_rate'])
            successful_f, successful_cr, improvements = ([], [], [])
            next_pop, next_fitness = (self.pop.copy(), self.fitness.copy())
            evaluated = min(n, self.budget - self.objective.fes)
            evaluated_points = list(self.iter_evaluate(trials[:evaluated]))
            for i, (genes, score) in enumerate(evaluated_points):
                if score < self.fitness[i]:
                    self.add_archive(self.pop[i], capacity)
                    successful_f.append(fs[i])
                    successful_cr.append(crs[i])
                    improvements.append(float(self.fitness[i] - score))
                if score <= self.fitness[i]:
                    next_pop[i], next_fitness[i] = (genes, score)
            self.pop, self.fitness = (next_pop, next_fitness)
            self.update_memory(successful_f, successful_cr, improvements)
            planned = max(4, round_positive(self.initial_n + (4 - self.initial_n) * self.objective.fes / self.budget))
            if planned < n:
                keep = np.argsort(self.fitness, kind='stable')[:planned]
                self.pop, self.fitness = (self.pop[keep], self.fitness[keep])
                capacity = int(planned * self.settings['archive_rate'])
                self.archive = self.archive[:capacity]
                self.current_p = p_rate(self.name, self.objective.fes / self.budget)
            self.history.append(dict(generation=self.generation, fe=self.objective.fes, population_size=len(self.pop), archive_size=len(self.archive), generation_start_fraction=fraction, generation_pbest_rate=generation_p, next_pbest_rate=self.current_p, evaluated=evaluated, partial_generation=evaluated < n, successes=len(successful_f), best_objective=self.best, MF=self.mf.tolist(), MCR=self.mcr.tolist()))
        if self.objective.fes != self.budget:
            raise RuntimeError('DE budget mismatch')

    def diagnostics(self):
        return dict(**self.settings, completed_fe=self.objective.fes, generations=self.generation, current_population=len(self.pop), archive_size=len(self.archive), repaired_candidates=self.repaired, history=self.history)
