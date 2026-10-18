import math
import numpy as np

VARIANTS = ("RM", "CM", "SW50", "SW60")
SWITCH = {"RM": None, "CM": None, "SW50": .5, "SW60": .6}

def population_diversity(population):
    pop = np.asarray(population, dtype=float)
    if len(pop) <= 1:
        return 0.
    diagonal = float(np.linalg.norm(np.full(pop.shape[1], 2.)))
    return float(np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1)) / diagonal)

def mutation_rate_multiset(n, diversity_ratio, mr_min, mr_max):
    ratio = float(np.clip(diversity_ratio, .1, 1.))
    adaptive_base = mr_min * (1.5 - .5 * ratio)
    levels = np.linspace(1., 0., int(n))
    return np.clip(adaptive_base * np.exp(levels), mr_min, mr_max)

def assign_mutation_rates(objectives, assignment, diversity_ratio, mr_min, mr_max):
    values = np.asarray(objectives, dtype=float)
    levels = mutation_rate_multiset(len(values), diversity_ratio, mr_min, mr_max)
    order = np.argsort(values, kind="stable")
    assigned = np.empty(len(values), dtype=float)
    assigned[order] = levels if assignment == "reverse" else levels[::-1]
    return assigned

def roulette_probabilities(objectives, bias):
    values = np.asarray(objectives, dtype=float)
    weights = float(np.max(values)) - values + float(bias)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full(len(values), 1. / len(values))
    p = weights / total
    return p / np.sum(p)

def levy_steps(rng, shape, beta=1.5):
    numerator = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    denominator = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (numerator / denominator) ** (1 / beta)
    u = rng.normal(0., sigma_u, size=shape)
    v = rng.normal(0., 1., size=shape)
    return u / np.power(np.maximum(np.abs(v), 1e-15), 1 / beta)

class ControlledHGA:
    def __init__(self, variant, objective, seed, budget, population_size=10,
                 max_generations=400, stagnation_generations=10,
                 mr_min=.05, mr_max=.30, cr=.80, mutation_scale=.10,
                 guided_restart_fraction=.40, levy_scale=.01, bias=1e-12):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.objective, self.budget = variant, objective, budget
        self.rng = np.random.default_rng(seed)
        self.n = population_size
        self.max_generations, self.T = max_generations, stagnation_generations
        self.mr_min, self.mr_max, self.cr = mr_min, mr_max, cr
        self.mutation_scale, self.guided_fraction = mutation_scale, guided_restart_fraction
        self.levy_scale, self.bias = levy_scale, bias
        self.generation = 0
        self.stagnation_counter = 0
        self.maintenance_events = 0
        self.switch_fe_actual = None
        self.switch_generation = None
        self.history = []
        self.population = None
        self.best_score = float('inf')
        self.best_genes = None
        self.boundary_hook = None

    def evaluate(self, proposal):

        raw = np.asarray(proposal, dtype=float).reshape(8)
        genes, _, _ = self.objective.resolve(raw, True)
        value = float(self.objective(raw))
        if value < self.best_score - 1e-15 or self.best_genes is None:
            self.best_score = value
            self.best_genes = genes.astype(float).copy()
        return genes.astype(float), value

    def initialize(self):
        proposals = self.rng.uniform(0., 2., size=(self.n, 8))
        self.population = np.zeros((self.n, 8), dtype=float)
        self.objectives = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            self.population[i], self.objectives[i] = self.evaluate(proposals[i])
        self.initial_diversity = max(population_diversity(self.population), 1e-15)

    def effective_assignment(self):
        fraction = SWITCH[self.variant]
        if fraction is None:
            return 'reverse' if self.variant == 'RM' else 'conventional'
        if self.objective.fes / self.budget < fraction:
            return 'reverse'
        if self.switch_fe_actual is None:
            self.switch_fe_actual = self.objective.fes
            self.switch_generation = self.generation
        return 'conventional'

    def create_offspring(self, rates, probabilities):
        pop = self.population
        n, d = pop.shape
        uniforms = self.rng.random(n)
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.
        indices = np.searchsorted(cumulative, uniforms, side='right')
        mates = pop[indices][self.rng.permutation(n)]
        chosen = pop[indices]
        crossover = self.rng.random((n, d)) < self.cr
        children = np.where(crossover, mates, chosen)
        selected_rates = rates[indices]
        mutation = self.rng.random((n, d)) < selected_rates[:, None]
        steps = self.rng.normal(0., 1., size=(n, d))
        children = children + mutation * steps * self.mutation_scale * 2.
        return np.clip(children, 0., 2.)

    def maintenance_population(self):
        n, d = self.population.shape
        count = min(max(int(round(n * self.guided_fraction)), 0), n)
        guided = self.best_genes + levy_steps(self.rng, (n, d)) * self.levy_scale * 2.
        random_points = self.rng.uniform(0., 2., size=(n, d))
        mask = np.arange(n)[:, None] < count
        return np.clip(np.where(mask, guided, random_points), 0., 2.)

    def run(self):
        if self.population is None:
            self.initialize()
            if self.boundary_hook:self.boundary_hook(self)
        while self.objective.fes < self.budget and self.generation < self.max_generations:
            self.generation += 1
            before = self.best_score
            diversity = population_diversity(self.population)
            ratio = float(np.clip(diversity / self.initial_diversity, .1, 1.))
            assignment = self.effective_assignment()
            rates = assign_mutation_rates(self.objectives, assignment, ratio, self.mr_min, self.mr_max)
            probs = roulette_probabilities(self.objectives, self.bias)
            children = self.create_offspring(rates, probs)
            maintenance = False
            if self.T > 0 and self.stagnation_counter >= self.T:
                children = self.maintenance_population()
                maintenance = True
                self.maintenance_events += 1
                self.stagnation_counter = 0
            elite = self.best_genes.copy()
            children[0] = elite
            next_pop = np.full_like(children, np.nan)
            next_obj = np.full(self.n, np.nan)
            next_pop[0], next_obj[0] = elite, self.best_score
            slots = list(range(1, self.n))[:self.budget-self.objective.fes]
            for i in slots:
                next_pop[i], next_obj[i] = self.evaluate(children[i])
            partial = bool(np.any(~np.isfinite(next_obj)))
            if not partial:
                self.population, self.objectives = next_pop, next_obj
                improved = self.best_score < before - 1e-15
                self.stagnation_counter = 0 if improved else self.stagnation_counter + 1
            self.history.append(dict(generation=self.generation, fe=self.objective.fes,
                                     assignment=assignment, maintenance=int(maintenance),
                                     stagnation_counter=self.stagnation_counter,
                                     switch_fe_actual=self.switch_fe_actual, partial=int(partial),
                                     best_objective=self.best_score))
            if not partial and self.boundary_hook:self.boundary_hook(self)
            if partial:
                break
        if self.objective.fes < self.budget:
            raise RuntimeError('HGA reached the generation limit before exhausting MAXFE')

    def diagnostics(self):
        return dict(variant=self.variant, fe=self.objective.fes, generations=self.generation,
                    stagnation_threshold=self.T, maintenance_events=self.maintenance_events,
                    switch_fe_actual=self.switch_fe_actual,
                    switch_generation=self.switch_generation, history=self.history)
