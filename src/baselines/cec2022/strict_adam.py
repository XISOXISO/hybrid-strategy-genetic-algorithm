from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

class StrictBudgetAdamHPO:

    def __init__(self, bounds, objective_function, minimize=True, random_state=None, verbose=False, budget=1000, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-08, grad_eps=1e-05, **kwargs):
        self.objective_func = objective_function
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = int(self.bounds.shape[0])
        self.minimize = bool(minimize)
        self.random_state = random_state
        self.verbose = verbose
        self.budget = int(budget)
        if self.budget < 1:
            raise ValueError('AdamHPO budget must be >= 1')
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.grad_eps = float(grad_eps)
        self.m = np.zeros(self.dim, dtype=np.float64)
        self.v = np.zeros(self.dim, dtype=np.float64)
        self.t = 0
        self.rng = np.random.RandomState(random_state)
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self.tail_policy_used = False
        self.tail_probe_count = 0

    def _record(self, p, v, bp, bv):
        self.history['params'].append(np.asarray(p, dtype=np.float64).tolist())
        self.history['values'].append(float(v))
        self.history['best_params'].append(np.asarray(bp, dtype=np.float64).tolist())
        self.history['best_values'].append(float(bv))

    def _is_better(self, a, b):
        return a < b if self.minimize else a > b

    def _adam_update(self, curr, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return np.clip(curr - step, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        curr = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.dim).astype(np.float64)
        curr_value = float(self.objective_func(curr))
        best = curr.copy()
        best_value = curr_value
        self._record(curr, curr_value, best, best_value)
        used = 1
        full_cost = self.dim + 1
        coord_offset = 0
        while self.budget - used >= full_cost:
            grad = np.zeros(self.dim, dtype=np.float64)
            for d in range(self.dim):
                p = curr.copy()
                p[d] = min(self.bounds[d, 1], p[d] + self.grad_eps)
                val = float(self.objective_func(p))
                used += 1
                if self._is_better(val, best_value):
                    best_value = val
                    best = p.copy()
                g = (val - curr_value) / self.grad_eps
                grad[d] = g if self.minimize else -g
            curr = self._adam_update(curr, grad)
            curr_value = float(self.objective_func(curr))
            used += 1
            if self._is_better(curr_value, best_value):
                best_value = curr_value
                best = curr.copy()
            self._record(curr, curr_value, best, best_value)
            coord_offset = (coord_offset + 1) % self.dim
        rem = self.budget - used
        if rem > 0:
            self.tail_policy_used = True
            if rem == 1:
                d = coord_offset % self.dim
                p = curr.copy()
                p[d] = min(self.bounds[d, 1], p[d] + self.grad_eps)
                val = float(self.objective_func(p))
                used += 1
                self.tail_probe_count = 1
                if self._is_better(val, best_value):
                    best_value = val
                    best = p.copy()
                self._record(curr, curr_value, best, best_value)
            else:
                m = min(self.dim, rem - 1)
                grad = np.zeros(self.dim, dtype=np.float64)
                for j in range(m):
                    d = (coord_offset + j) % self.dim
                    p = curr.copy()
                    p[d] = min(self.bounds[d, 1], p[d] + self.grad_eps)
                    val = float(self.objective_func(p))
                    used += 1
                    if self._is_better(val, best_value):
                        best_value = val
                        best = p.copy()
                    g = (val - curr_value) / self.grad_eps
                    grad[d] = g if self.minimize else -g
                self.tail_probe_count = m
                curr = self._adam_update(curr, grad)
                curr_value = float(self.objective_func(curr))
                used += 1
                if self._is_better(curr_value, best_value):
                    best_value = curr_value
                    best = curr.copy()
                self._record(curr, curr_value, best, best_value)
        if used != self.budget:
            raise RuntimeError(f'StrictBudgetAdamHPO internal FE mismatch: {used} != {self.budget}')
        return {'best_value': float(best_value), 'best_params': best.tolist(), 'history': self.history, 'tail_policy_used': self.tail_policy_used, 'tail_probe_count': self.tail_probe_count}

    def get_optimization_history(self):
        return self.history

    def get_best_result(self):
        if not self.history['best_values']:
            return (None, float('inf') if self.minimize else float('-inf'))
        return (self.history['best_params'][-1], self.history['best_values'][-1])
