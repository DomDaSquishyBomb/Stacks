# Mathing
import numpy as np
from scipy.stats import norm
# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

from Gershman2017 import KalmanFilter, ContextFree, AdditiveContext, ModulatoryContext, GershmanModel
from Stacks import StackedModel

#---Experiment replication---

def run_2017_experiment(model = GershmanModel):
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
        m = model(n_cues=3, n_contexts=3)
        for (x, ctx, r) in schedule:
            m.update(x, ctx, r)
        posteriors[condition] = m.posterior
        results[condition] = {
            label: m.predict(x, ctx)["p_outcome"]
            for (x, ctx, label) in test_trials
        }
    return results, posteriors

def run_RenewalABC_experiment(model = StackedModel):
    x1 = np.array([1])

    schedules = {
        "Reward First Training": (
            [(x1, 0, 1)] * 5 + 
            [(x1, 1, 0)] * 5 
        ),
        "Punish First Training": (
            [(x1, 1, 0)] * 5 + 
            [(x1, 0, 1)] * 5 
        ),
    }

    test_trials = [
        (x1, 2, "x1c3"),
    ]

    results, posteriors, stack_compositions = {}, {}, {}
    for condition, schedule in schedules.items():
        m = model(n_cues=1, n_contexts=3)
        for (x, ctx, r) in schedule:
            m.update(x, ctx, r)
        posteriors[condition] = m.posterior
        stack_compositions[condition] = m.stack_composition if hasattr(m, "stack_composition") else None
        results[condition] = {
            label: m.predict(x, ctx)["p_outcome"]
            for (x, ctx, label) in test_trials
        }
    return results, posteriors, stack_compositions

def run_RenewalABA_experiment(model = StackedModel):
    x1 = np.array([1])

    schedules = {
        "Reward First Training": (
            [(x1, 0, 1)] * 5 + 
            [(x1, 1, 0)] * 5 
        ),
        "Punish First Training": (
            [(x1, 1, 0)] * 5 + 
            [(x1, 0, 1)] * 5 
        ),
    }

    test_trials = {
        "Reward First Training": [(x1, 0, "x1c1")],
        "Punish First Training": [(x1, 1, "x1c1")],
    }

    results, posteriors, stack_compositions = {}, {}, {}
    for condition, schedule in schedules.items():
        m = model(n_cues=1, n_contexts=3)
        for (x, ctx, r) in schedule:
            m.update(x, ctx, r)
        posteriors[condition] = m.posterior
        stack_compositions[condition] = m.stack_composition if hasattr(m, "stack_composition") else None
        results[condition] = {
            label: m.predict(x, ctx)["p_outcome"]
            for (x, ctx, label) in test_trials[condition]
        }
    return results, posteriors, stack_compositions

#---Plotting---

def plot_gershman_results(all_results, all_posteriors, name="gershman_results"):
    model_names = GershmanModel.MODELS
    conditions = list(all_posteriors[0].keys())

    post_df = {
        "Training Group": [cond for posteriors in all_posteriors for cond in conditions for _ in model_names],
        "Model": [name for _ in all_posteriors for _ in conditions for name in model_names],
        "Posterior": [posteriors[cond][i] for posteriors in all_posteriors for cond in conditions for i in range(len(model_names))],
    }

    trials = list(next(iter(all_results[0].values())).keys())
    test_df = {
        "Training Group": [cond for results in all_results for cond in conditions for _ in trials],
        "Test Condition": [trial for _ in all_results for _ in conditions for trial in trials],
        "P(outcome)": [results[cond][trial] for results in all_results for cond in conditions for trial in trials],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.barplot(data=post_df, x="Training Group", y="Posterior", hue="Model", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Posterior over structures after training")
    axes[0].legend(fontsize=8)

    sns.barplot(data=test_df, x="Training Group", y="P(outcome)", hue="Test Condition", ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test phase predictions")
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=150)
    plt.close()

def plot_stacked_results(all_results, all_stack_compositions, name = "stacked_results"):
    model_names = StackedModel.MODELS
    conditions = list(all_stack_compositions[0].keys())

    stack_df = {
        "Training Group": [cond for sc in all_stack_compositions for cond in conditions for _ in model_names],
        "Model": [name for _ in all_stack_compositions for _ in conditions for name in model_names],
        "Proportion": [sc[cond][i] for sc in all_stack_compositions for cond in conditions for i in range(len(model_names))],
    }

    trials = list(next(iter(all_results[0].values())).keys())
    test_df = {
        "Training Group": [cond for results in all_results for cond in conditions for _ in trials],
        "Test Condition": [trial for _ in all_results for _ in conditions for trial in trials],
        "P(outcome)": [results[cond][trial] for results in all_results for cond in conditions for trial in trials],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.barplot(data=stack_df, x="Training Group", y="Proportion", hue="Model", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Stack composition after training")
    axes[0].legend(fontsize=8)

    sns.barplot(data=test_df, x="Training Group", y="P(outcome)", hue="Test Condition", ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test phase predictions")
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(name, dpi=150)
    plt.close()


def plot_staged_behaviour(model=StackedModel, n_sims=100, name="staged_behaviour"):
    x1 = np.array([1])
    conditions = ["Reward First Training", "Punish First Training"]

    schedules = {
        "Reward First Training": [(x1, 0, 1)] * 5 + [(x1, 1, 0)] * 5,
        "Punish First Training": [(x1, 1, 0)] * 5 + [(x1, 0, 1)] * 5,
    }
    renewal_tests = {
        "ABA": {
            "Reward First Training": (x1, 0),
            "Punish First Training": (x1, 1),
        },
        "ABC": {
            "Reward First Training": (x1, 2),
            "Punish First Training": (x1, 2),
        },
    }

    records = []
    for _ in range(n_sims):
        for condition, schedule in schedules.items():
            m = model(n_cues=1, n_contexts=3)
            for t, (x, ctx, r) in enumerate(schedule):
                m.update(x, ctx, r)
                pred = m.predict(x, ctx)
                for renewal in renewal_tests:
                    records.append({
                        "renewal": renewal,
                        "condition": condition,
                        "trial": t + 1,
                        "model": pred["active_model"],
                        "p_outcome": pred["p_outcome"],
                    })
            for renewal, test_trials in renewal_tests.items():
                x_test, ctx_test = test_trials[condition]
                pred = m.predict(x_test, ctx_test)
                records.append({
                    "renewal": renewal,
                    "condition": condition,
                    "trial": 11,
                    "model": pred["active_model"],
                    "p_outcome": pred["p_outcome"],
                })

    model_names = ["ContextFree", "ModulatoryContext", "AdditiveContext"]
    renewals = ["ABA", "ABC"]
    trials = list(range(1, 12))
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    for row_pair, renewal in enumerate(renewals):
        for col, condition in enumerate(conditions):
            cond = [r for r in records if r["renewal"] == renewal and r["condition"] == condition]

            # --- p_outcome over trials ---
            mean_p = [np.mean([r["p_outcome"] for r in cond if r["trial"] == t]) for t in trials]
            sem_p  = [np.std( [r["p_outcome"] for r in cond if r["trial"] == t]) / np.sqrt(n_sims) for t in trials]

            ax = axes[row_pair * 2, col]
            ax.errorbar(trials, mean_p, yerr=sem_p, marker="o", capsize=3)
            ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(5.5,  color="gray", linestyle=":", linewidth=0.8)
            ax.axvline(10.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xticks(trials)
            ax.set_xticklabels([*range(1, 11), "Test"])
            ax.set_ylim(0, 1)
            ax.set_title(f"{renewal} — {condition}\nP(outcome) over trials")
            ax.set_ylabel("P(outcome)")
            for x_pos, label in [(3, "Phase A"), (8, "Phase B"), (11, "Test")]:
                ax.text(x_pos, 0.95, label, ha="center", fontsize=7, color="gray")

            # --- active model proportion over trials ---
            ax = axes[row_pair * 2 + 1, col]
            bottom = np.zeros(len(trials))
            for mname in model_names:
                props = [np.mean([r["model"] == mname for r in cond if r["trial"] == t]) for t in trials]
                ax.bar(trials, props, bottom=bottom, label=mname)
                bottom += np.array(props)
            ax.axvline(5.5,  color="gray", linestyle=":", linewidth=0.8)
            ax.axvline(10.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xticks(trials)
            ax.set_xticklabels([*range(1, 11), "Test"])
            ax.set_ylim(0, 1)
            ax.set_title("Active model proportion")
            ax.set_ylabel("Proportion")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=150)
    plt.close()
    print(f"Plot saved to {name}.png")


if __name__ == "__main__":
    n_sims = 100
    
    
    ### 2017 Experiment Replication ###
    print(f"2017 Experiment Results ({n_sims} simulations)\n")

    print("=== Gershman 2017 ===")
    all_results_g, all_posteriors_g = [], []
    for _ in range(n_sims):
        results, posteriors = run_2017_experiment()
        all_results_g.append(results)
        all_posteriors_g.append(posteriors)
    plot_gershman_results(all_results_g, all_posteriors_g, name="gershman2017_results")
    print("Plot saved to gershman2017_results.png\n")

    ### Current Experiment: ABC Renewal ###
    print(f"ABC Renewal Experiment Results ({n_sims} simulations)\n")

    print("=== Gershman Curr ===")
    all_results_g, all_posteriors_g = [], []
    for _ in range(n_sims):
        results, posteriors, _ = run_RenewalABC_experiment(model=GershmanModel)
        all_results_g.append(results)
        all_posteriors_g.append(posteriors)
    plot_gershman_results(all_results_g, all_posteriors_g, name="gershmanABC_results")
    print("Plot saved to gershmanABC_results.png\n")

    print("=== Stack Curr ===")
    all_results_s, all_stack_compositions_s = [], []
    for _ in range(n_sims):
        results, _, stack_compositions = run_RenewalABC_experiment(model=StackedModel)
        all_results_s.append(results)
        all_stack_compositions_s.append(stack_compositions)
    plot_stacked_results(all_results_s, all_stack_compositions_s, name="stackedABC_results")
    print("Plot saved to stackedABC_results.png\n")
    
    ### Current Experiment: ABA Renewal ###
    print(f"ABA Renewal Experiment Results ({n_sims} simulations)\n")

    print("=== Gershman Curr ===")
    all_results_g, all_posteriors_g = [], []
    for _ in range(n_sims):
        results, posteriors, _ = run_RenewalABA_experiment(model=GershmanModel)
        all_results_g.append(results)
        all_posteriors_g.append(posteriors)
    plot_gershman_results(all_results_g, all_posteriors_g, name="gershmanABA_results")
    print("Plot saved to gershmanABA_results.png\n")

    print("=== Stack Curr ===")
    all_results_s, all_stack_compositions_s = [], []
    for _ in range(n_sims):
        results, _, stack_compositions = run_RenewalABA_experiment(model=StackedModel)
        all_results_s.append(results)
        all_stack_compositions_s.append(stack_compositions)
    plot_stacked_results(all_results_s, all_stack_compositions_s, name="stackedABA_results")
    print("Plot saved to stackedABA_results.png\n")

    ### Staged Behaviour ###
    print("=== Model Pick at Every Stage ===")
    plot_staged_behaviour(model=StackedModel, n_sims=n_sims, name="models_at_stages")


