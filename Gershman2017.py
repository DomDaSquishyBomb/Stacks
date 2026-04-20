# Mathing
import numpy as np
from scipy.stats import norm
# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

#---Global Parameters---
SIG_SQRD_R = 0.01 # variance of reward noise
SIG_SQRD_W = 1.0 # variance of prior weight noise
TAU_SQRD = 0.001 # variance of learning noise

#---Kalman Filter---
class KalmanFilter:
    # In house implementation to reduce complexities.
    def __init__(self, dim: int, tau_sqrd: float = TAU_SQRD):
        # Based on parameters from appendix of Gershman 2017
        self.w_hat = np.zeros(dim)
        self.Sig = SIG_SQRD_W * np.eye(dim)
        self.tau_sqrd = tau_sqrd

    def diffuse(self):
        # Random walk diffusion of the weights
        self.Sig += self.tau_sqrd * np.eye(len(self.w_hat))
        
    def _innovation(self, x: np.ndarray) -> tuple[float, float]:
        # Compute the difference between predicted and actual reward, and the variance of this prediction
        mu = float(x @ self.w_hat)
        var = float(x.T @ self.Sig @ x + SIG_SQRD_R)
        return mu, var
    
    def log_likelihood(self, x: np.ndarray, r: float) -> float:
        # Compute the log likelihood of observing reward r given the current model
        mu, var = self._innovation(x)
        return float(norm.logpdf(r, loc=mu, scale=np.sqrt(var)))
    
    def update(self, x: np.ndarray, r: float):
        # Update the weights and covariance based on the observed reward
        mu, var = self._innovation(x)
        g = self.Sig @ x / var  # Kalman gain
        self.w_hat += g * (r - mu)  # Update weights
        self.Sig -= np.outer(g, x) @ self.Sig  # Update covariance
        
    def predict(self, x: np.ndarray) -> float:
        # Predict the reward for a given input
        return float(x @ self.w_hat)

#---Three Causal Structures---
class CausalStructure:
    def log_likelihood(self, x: np.ndarray, context: int, r: float) -> float:
        raise NotImplementedError
    def update(self, x: np.ndarray, context: int, r: float):
        raise NotImplementedError
    def predict(self, x: np.ndarray, context: int) -> float:
        raise NotImplementedError
    def get_complexity(self) -> int:
        raise NotImplementedError
    def is_novel(self, context: int) -> bool:
        return False

class ContextFree(CausalStructure):
    def __init__(self, n_cues: int, n_contexts: int):
        self.n_cues = n_cues
        self.kf = KalmanFilter(n_cues)
    
    def log_likelihood(self, x, context, r):
        return self.kf.log_likelihood(x, r)
    
    def update(self, x, context, r):
        self.kf.diffuse()
        self.kf.update(x, r)
    
    def predict(self, x, context):
        return self.kf.predict(x)
    
    def get_complexity(self) -> int:
        return self.n_cues 

class ModulatoryContext(CausalStructure):
    def __init__(self, n_cues: int, n_contexts: int):
        self.n_cues = n_cues
        self.kfs: dict[int, KalmanFilter] = {}

    def _get(self, context: int) -> KalmanFilter:
        if context not in self.kfs:
            self.kfs[context] = KalmanFilter(self.n_cues)
        return self.kfs[context]

    def is_novel(self, context: int) -> bool:
        return context not in self.kfs

    def log_likelihood(self, x, context, r):
        return self._get(context).log_likelihood(x, r)

    def update(self, x, context, r):
        kf = self._get(context)
        kf.diffuse()
        kf.update(x, r)

    def predict(self, x, context):
        if context not in self.kfs:
            return 0.0  # prior mean — don't create a KF just from predicting
        return self.kfs[context].predict(x)
    
    def get_complexity(self) -> int:
        return (len(self.kfs) + 1) * self.n_cues
    
class AdditiveContext(CausalStructure):
    def __init__(self, n_cues: int, n_contexts: int):
        self.n_cues     = n_cues
        self.n_contexts = n_contexts
        self.kf         = KalmanFilter(n_cues + n_contexts)

    def _augment(self, x: np.ndarray, context: int) -> np.ndarray:
        c = np.zeros(self.n_contexts)
        if context < self.n_contexts:
            c[context] = 1.0
        return np.concatenate([x, c])

    def log_likelihood(self, x, context, r):
        return self.kf.log_likelihood(self._augment(x, context), r)

    def update(self, x, context, r):
        self.kf.diffuse()
        self.kf.update(self._augment(x, context), r)

    def predict(self, x, context):
        return self.kf.predict(self._augment(x, context))
    
    def get_complexity(self) -> int:
        return self.n_cues + self.n_contexts
    
#--- Main ---

class GershmanModel:
    """
    Parameters:
    - n_cues: Number of cue dimensions
    - n_contexts: Number of contexts
    """
    
    MODELS = ["ContextFree", "ModulatoryContext", "AdditiveContext"]
    
    def __init__(self, n_cues: int, n_contexts: int = 10):
        self.structures = [ContextFree(n_cues, n_contexts), ModulatoryContext(n_cues, n_contexts), AdditiveContext(n_cues, n_contexts)]
        self.log_priors = np.zeros(len(self.structures))  # Log prior over models, initialized to uniform
        self._history = []  # To store history of predictions and updates for analysis
        
    @property
    def posterior(self) -> np.ndarray:
        lp = self.log_priors - self.log_priors.max()  # For numerical stability
        p = np.exp(lp)
        return p / p.sum()  # Normalize to get posterior probabilities
        
    def update(self, x: np.ndarray, context: int, outcome: float):
        # Update each model and compute log likelihoods
        x = np.asarray(x, dtype=float)
        for i, m in enumerate(self.structures):
            self.log_priors[i] += m.log_likelihood(x, context, outcome)
        for m in self.structures:
            m.update(x, context, outcome)
        post = self.posterior
        self._history.append({"x": x, "context": context, "outcome": outcome, "posterior": post.copy()})
        return post
    
    def predict(self, x: np.ndarray, context: int) -> float:
        x = np.asarray(x, dtype=float)
        post = self.posterior
        Vs = [m.predict(x, context) for m in self.structures]
        V = float(np.dot(post, Vs))  # Model averaging
        p = 1.0 / (1.0 + np.exp(-2 * V + 1))  # Softmax to get choice probability
        return {"V": V, "p_outcome": p, "per_model": dict(zip(self.MODELS, Vs))}
    
    def posterior_history(self):
        if not self._history:
            return None
        return np.stack([h["posterior"] for h in self._history])