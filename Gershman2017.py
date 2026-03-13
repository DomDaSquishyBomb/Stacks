# Mathing
import numpy as np
from scipy.stats import norm
# Graphing
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import seaborn as sns

#---Global Parameters---
SIG_SQRD_R = 0.01 # variance of reward noise
SIG_SQRD_W = 1.0 # variance of prior weight noise
TAU_SQRD = 0.001 # variance of learning noise

#---Kalman Filter---
class KalmanFilter:
    # In house implementation to reduce complexities.
    def __init__(self, dim: int):
        # Based on parameters from appendix of Gershman 2017
        self.w_hat = np.zeros(dim)
        self.Sig = SIG_SQRD_W * np.eye(dim)
        
    def diffuse(self):
        # Random walk diffusion of the weights
        self.Sig += TAU_SQRD * np.eye(len(self.w_hat))
        
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
class ContextFree():
    def __init__(self, n_cues: int, n_contexts: int):
        self.kf = KalmanFilter(n_cues)
    
    def log_likelihood(self, x, context, r):
        return self.kf.log_likelihood(x, r)
    
    def update(self, x, context, r):
        self.kf.diffuse()
        self.kf.update(x, r)
    
    def predict(self, x, context):
        return self.kf.predict(x)

class ModulatoryContext:
    def __init__(self, n_cues: int, n_contexts: int):
        self.n_cues = n_cues
        self.kfs: dict[int, KalmanFilter] = {}

    def _get(self, context: int) -> KalmanFilter:
        if context not in self.kfs:
            self.kfs[context] = KalmanFilter(self.n_cues)
        return self.kfs[context]

    def log_likelihood(self, x, context, r):
        return self._get(context).log_likelihood(x, r)

    def update(self, x, context, r):
        kf = self._get(context)
        kf.diffuse()
        kf.update(x, r)

    def predict(self, x, context):
        return self._get(context).predict(x)
    
class AdditiveContext:
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

class StackedModel:
    """
    Parameters:
    - n_cues: Number of cue dimensions
    - n_contexts: Number of contexts
    """
    def __init__(self, n_cues: int, n_contexts: int = 10):
        pass
        
    def update(self, x: np.ndarray, context: int, outcome: float):
        pass
    
    def predict(self, x: np.ndarray, context: int) -> dict[str, float]:
        pass
    
    def posterior_history(self):
        pass


#---Experiment replication---

def run_experiment():
    x1 = np.array([1, 0, 0])
    x2 = np.array([0, 1, 0])
    x3 = np.array([0, 0, 1])

    schedules = {
        "Irrelevant Training": (
            [(x1, 0, 1), (x2, 0, 0)] * 5 +
            [(x1, 1, 1), (x2, 1, 0)] * 5
        ),
        "Modulatory Training": (
            [(x1, 0, 1), (x2, 0, 0)] * 5 +
            [(x1, 1, 0), (x2, 1, 1)] * 5
        ),
        "Additive Training": (
            [(x1, 0, 1), (x2, 0, 1)] * 5 +
            [(x1, 1, 0), (x2, 1, 0)] * 5
        ),
    }

    test_trials = [
        (x1, 0, "x1c1"), (x1, 2, "x1c3"),
        (x3, 0, "x3c1"), (x3, 2, "x3c3"),
    ]

    results, posteriors = {}, {}
    for condition, schedule in schedules.items():
        model = GershmanModel(n_cues=3, n_contexts=3)
        for (x, ctx, r) in schedule:
            model.update(x, ctx, r)
        posteriors[condition] = model.posterior
        results[condition] = {
            label: model.predict(x, ctx)["p_outcome"]
            for (x, ctx, label) in test_trials
        }
    return results, posteriors


def plot_results(results, posteriors):
    conditions = list(posteriors.keys())

    # Build data for seaborn as dicts of lists
    post_df = {
        "Training Group": [cond for cond in conditions for _ in GershmanModel.MODELS],
        "Model": [name for _ in conditions for name in GershmanModel.MODELS],
        "Posterior": [posteriors[cond][i] for cond in conditions for i in range(len(GershmanModel.MODELS))],
    }

    trials = list(next(iter(results.values())).keys())
    test_df = {
        "Training Group": [cond for cond in conditions for _ in trials],
        "Test Condition": [trial for _ in conditions for trial in trials],
        "P(outcome)": [results[cond][trial] for cond in conditions for trial in trials],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.barplot(data=post_df, x="Training Group", y="Posterior", hue="Model",
                ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Model posterior after training")
    axes[0].legend(fontsize=8)

    sns.barplot(data=test_df, x="Training Group", y="P(outcome)", hue="Test Condition",
                ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test phase predictions")
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("gershman2017_results.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    results, posteriors = run_experiment()

    print("Posterior over structures after training:")
    for cond, post in posteriors.items():
        print(f"  {cond:12s}  M1={post[0]:.3f}  M2={post[1]:.3f}  M3={post[2]:.3f}")

    print("\nTest phase predictions P(outcome):")
    header = f"{'Trial':<10}" + "".join(f"{c:<14}" for c in results)
    print(header)
    for trial in list(next(iter(results.values())).keys()):
        row = f"{trial:<10}" + "".join(f"{results[c][trial]:<14.3f}" for c in results)
        print(row)

    plot_results(results, posteriors)
    print("\nPlot saved.")
