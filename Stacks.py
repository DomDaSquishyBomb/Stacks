# Mathing
import numpy as np
from scipy.stats import norm
# Graphing
import matplotlib.pyplot as plt
import seaborn as sns
# Stacking
from collections import deque

#---Global Parameters---
SIG_SQRD_R = 0.01 # variance of reward noise
SIG_SQRD_W = 1.0 # variance of prior weight noise
TAU_SQRD = 0.001 # variance of learning noise

from Gershman2017 import KalmanFilter, ContextFree, AdditiveContext, ModulatoryContext, GershmanModel, CausalStructure

class StackedModel:
    """
    Parameters:
    - n_cues: Number of cue dimensions
    - n_contexts: Number of contexts
    """
    
    MODELS = ["ContextFree", "ModulatoryContext", "AdditiveContext"]
    
    def __init__(self, n_cues: int, n_contexts: int = 10, strategy = "different simplest", time = 2, threshold = 0.5, spread = 0.2):
        self.structures = [ContextFree(n_cues, n_contexts), ModulatoryContext(n_cues, n_contexts), AdditiveContext(n_cues, n_contexts)]
        self.stack = []  # Stack of models currently in use
        self.log_priors = np.zeros(len(self.structures))  # Log prior over models, initialized to uniform
        self._history = []  # To store history of predictions and updates for analysis
        self.strategy = strategy
        self.time = time
        self.threshold = threshold
        self.spread = spread  # 0 = only top model updates, 1 = all update equally
        
    @property
    def posterior(self) -> np.ndarray:
        lp = self.log_priors - self.log_priors.max()  # For numerical stability
        p = np.exp(lp)
        return p / p.sum()  # Normalize to get posterior probabilities
        
    def update(self, x: np.ndarray, context: int, outcome: float):
        x = np.asarray(x, dtype=float)

        if not self.stack:
            self.model_select(strategy=self.strategy)
        else:
            pred = self.predict(x, context)
            mispredicted = abs(outcome - pred["p_outcome"]) > self.threshold
            if mispredicted:
                self.model_select(curr_model=self.stack[-1], strategy=self.strategy)

        # Compute exponential weights by stack depth (spread=0: top only, spread=1: all equal)
        # Stack members: weight = spread^depth (depth=0 for top)
        # Non-stack members: weight = spread^len(stack) (treated as beyond the stack)
        weights = {m: self.spread ** len(self.stack) for m in self.structures} 
        for depth, m in enumerate(reversed(self.stack)):
            weights[m] = self.spread ** depth
            
        ### Uncomment to not include weight outside of stack
        # weights = {m: 0.0 for m in self.structures}

        # Update each structure with probability = its weight
        for i, m in enumerate(self.structures):
            if np.random.random() < weights[m]:
                self.log_priors[i] += m.log_likelihood(x, context, outcome)
                m.update(x, context, outcome)
        post = self.posterior
        self._history.append({"x": x, "context": context, "outcome": outcome, "posterior": post.copy()})
        return post
    
    def predict(self, x: np.ndarray, context: int) -> float:
        x = np.asarray(x, dtype=float)
        curr = self.stack[-1]
        if curr.is_novel(context):
            for m in reversed(self.stack):
                if not m.is_novel(context):
                    curr = m
                    break
        V = curr.predict(x, context)
        p = 1.0 / (1.0 + np.exp(-2 * V + 1))  # Softmax to get choice probability
        per_model = {name: m.predict(x, context) for name, m in zip(self.MODELS, self.structures)}
        return {"V": V, "p_outcome": p, "per_model": per_model, "active_model": type(curr).__name__}
    
    @property
    def stack_composition(self) -> np.ndarray:
        counts = np.zeros(len(self.structures))
        for m in self.stack:
            counts[self.structures.index(m)] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    def posterior_history(self):
        if not self._history:
            return None
        return np.stack([h["posterior"] for h in self._history])
    
    def model_select(self, curr_model = None, n = 1, strategy = "simplest"):
        """
        Randomly selects n models that share variables with the situation, then picks the one with the lowest parameters.
        """
        structures = {m: m.get_complexity() for m in self.structures}
        
        for _ in range(n):
            if strategy == "simplest":
                pick = min(structures, key=structures.get)
        
            elif strategy == "random":
                pick = np.random.choice(list(structures.keys()))
                
            elif strategy == "weighted":
                inv = np.array([1 / v for v in structures.values()])
                pick = np.random.choice(list(structures.keys()), p = inv / inv.sum())
            
            elif strategy == "different":
                different_structures = structures.copy()
                different_structures.pop(curr_model, None)
                pick = np.random.choice(list(different_structures.keys()))
                
            elif strategy == "different simplest":
                different_structures = structures.copy()
                different_structures.pop(curr_model, None)
                pick = min(different_structures, key=different_structures.get)
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if pick in self.stack:
                self.stack.remove(pick)
                
            self.stack.append(pick)
            
        
            
    # def get_valid_models(self, situation):
    #     valid = []
    #     for m in self.MODELS:
    #         if m == "ContextFree":
    #             valid.append(ContextFree(self.n_cues, self.n_contexts))
    #         elif m == "ModulatoryContext":
    #             valid.append(ModulatoryContext(self.n_cues, self.n_contexts))
    #         elif m == "AdditiveContext":
    #             valid.append(AdditiveContext(self.n_cues, self.n_contexts))
    #     return valid
        
        