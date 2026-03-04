import argparse
import csv
from dataclasses import dataclass
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection

from poly_sbst.common.abstract_executor import AbstractExecutor
from poly_sbst.common.abstract_grammar import AbstractGrammar
from poly_sbst.common.random_seed import get_random_seed
from poly_sbst.crossover.url_test_suite_crossover import URLTestSuiteCrossover
from poly_sbst.generators.url_test_suite_generator import URLTestSuiteGenerator
from poly_sbst.mutation.url_test_suite_mutation import URLTestSuiteMutation
from poly_sbst.problems.url_test_suite_problem import URLTestSuiteProblem
from poly_sbst.sampling.abstract_sampling import AbstractSampling


@dataclass
class RunSummary:
    label: str
    best_ratio: float
    best_suite_size: int
    trace: np.ndarray


def build_url_grammar() -> AbstractGrammar:
    grammar = {
        "<start>": ["<url>"],
        "<url>": ["<scheme>://<authority><path><query><fragment>", "<path>", "<scheme>:<path>"],
        "<scheme>": ["http", "https", "ftp", "file", "mailto", "ws", "wss"],
        "<authority>": ["", "<host>", "<userinfo>@<host>", "<host>:<port>", "<userinfo>@<host>:<port>"],
        "<userinfo>": ["user", "admin", "test", "<alnum><alnum>"],
        "<host>": ["localhost", "example.com", "127.0.0.1", "[::1]", "sub.domain.org"],
        "<port>": ["80", "443", "8080", "65535", "0"],
        "<path>": ["", "/", "/<segment>", "/<segment>/<segment>", "/<segment>/<segment>/<segment>"],
        "<segment>": ["a", "api", "v1", "test", "..", "%2F", "<alnum><alnum>"],
        "<query>": ["", "?<pair>", "?<pair>&<pair>", "?<pair>&<pair>&<pair>"],
        "<pair>": ["k=v", "x=1", "q=test", "empty=", "encoded=%20", "<alnum>=<alnum>"],
        "<fragment>": ["", "#frag", "#", "#<alnum><alnum>"],
        "<alnum>": list("abcdefghijklmnopqrstuvwxyz0123456789"),
    }
    return AbstractGrammar(grammar)


def running_best_trace(problem: URLTestSuiteProblem) -> np.ndarray:
    if not problem.execution_data:
        return np.array([])

    ratios = np.array([problem.execution_data[i]["ratio"] for i in sorted(problem.execution_data.keys())], dtype=float)
    return np.maximum.accumulate(ratios)


def make_ga(problem, generator, selection_mode: str, pop_size: int):
    selection = (
        TournamentSelection(func_comp=comp_by_cv_and_fitness)
        if selection_mode == "tournament"
        else RandomSelection()
    )

    return GA(
        pop_size=pop_size,
        n_offsprings=max(2, int(pop_size / 2)),
        sampling=AbstractSampling(generator),
        mutation=URLTestSuiteMutation(generator=generator),
        crossover=URLTestSuiteCrossover(cross_rate=0.9, max_length=generator.max_length),
        selection=selection,
        eliminate_duplicates=False,
    )


def run_strategy(label: str, strategy: str, runs: int, budget: int, pop_size: int) -> list[RunSummary]:
    summaries = []

    for i in range(runs):
        print(f"Run {i + 1}/{runs} - Strategy: {label}")
        seed = get_random_seed()
        grammar = build_url_grammar()
        generator = URLTestSuiteGenerator(grammar)
        executor = AbstractExecutor(urlparse)
        problem = URLTestSuiteProblem(executor)

        if strategy == "random_search":
            method = RandomSearch(sampling=AbstractSampling(generator))
        else:
            method = make_ga(problem, generator, strategy, pop_size)

        res = minimize(
            problem,
            method,
            termination=("n_eval", budget),
            seed=seed,
            verbose=False,
            eliminate_duplicates=False,
            save_history=False,
        )

        trace = running_best_trace(problem)
        best_ratio = 0.0 if trace.size == 0 else float(trace[-1])

        if res.X is not None and len(res.X) > 0 and len(res.X[0]) > 0:
            best_suite_size = len(res.X[0])
        else:
            best_suite_size = 0

        summaries.append(
            RunSummary(
                label=label,
                best_ratio=best_ratio,
                best_suite_size=best_suite_size,
                trace=trace,
            )
        )

    return summaries


def pad_and_average(traces: list[np.ndarray]) -> np.ndarray:
    max_len = max((len(t) for t in traces), default=0)
    if max_len == 0:
        return np.array([])

    padded = []
    for trace in traces:
        if len(trace) == max_len:
            padded.append(trace)
        elif len(trace) == 0:
            padded.append(np.zeros(max_len))
        else:
            tail = np.full(max_len - len(trace), trace[-1])
            padded.append(np.concatenate([trace, tail]))

    return np.mean(np.vstack(padded), axis=0)


def plot_comparison(results: dict[str, list[RunSummary]], output_path: str):
    plt.figure(figsize=(10, 5))

    for label, summaries in results.items():
        traces = [s.trace for s in summaries]
        avg_trace = pad_and_average(traces)
        if len(avg_trace) == 0:
            continue

        x = np.arange(1, len(avg_trace) + 1)
        plt.plot(x, avg_trace, label=label)

    plt.title("URL: meilleure valeur cumulative de nl/nt par évaluation")
    plt.xlabel("Évaluation")
    plt.ylabel("Meilleur nl/nt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary(results: dict[str, list[RunSummary]]):
    print("=== URL Optimization Summary ===")
    for label, summaries in results.items():
        best_ratio = max((s.best_ratio for s in summaries), default=0.0)
        avg_ratio = float(np.mean([s.best_ratio for s in summaries])) if summaries else 0.0
        avg_size = float(np.mean([s.best_suite_size for s in summaries])) if summaries else 0.0
        print(
            f"{label}: best_max_nl_nt={best_ratio:.4f}, avg_best_nl_nt={avg_ratio:.4f}, avg_best_suite_size={avg_size:.2f}"
        )


def export_run_csv(results: dict[str, list[RunSummary]], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["strategy", "run", "best_ratio", "best_suite_size", "trace_length"])

        for label, summaries in results.items():
            for run_idx, summary in enumerate(summaries, start=1):
                writer.writerow([
                    label,
                    run_idx,
                    f"{summary.best_ratio:.8f}",
                    summary.best_suite_size,
                    len(summary.trace),
                ])


def export_trace_csv(results: dict[str, list[RunSummary]], output_path: str):
    avg_traces = {label: pad_and_average([s.trace for s in summaries]) for label, summaries in results.items()}
    max_len = max((len(trace) for trace in avg_traces.values()), default=0)

    with open(output_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["evaluation"] + list(avg_traces.keys()))

        for i in range(max_len):
            row = [i + 1]
            for label in avg_traces.keys():
                trace = avg_traces[label]
                if i < len(trace):
                    row.append(f"{trace[i]:.8f}")
                elif len(trace) > 0:
                    row.append(f"{trace[-1]:.8f}")
                else:
                    row.append("")
            writer.writerow(row)


def optimize_url(runs: int = 5, budget: int = 5000, pop_size: int = 50, plot_file: str = "url_comparison.png"):
    results = {
        "GA + TournamentSelection": run_strategy("GA + TournamentSelection", "tournament", runs, budget, pop_size),
        "GA + RandomSelection": run_strategy("GA + RandomSelection", "random", runs, budget, pop_size),
        "RandomSearch": run_strategy("RandomSearch", "random_search", runs, budget, pop_size),
    }

    plot_comparison(results, plot_file)
    export_run_csv(results, "url_results_runs.csv")
    export_trace_csv(results, "url_results_trace.csv")
    print_summary(results)
    print(f"Plot saved to: {plot_file}")
    print("CSV saved to: url_results_runs.csv, url_results_trace.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize URL test suites with GA/RandomSearch.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--plot", type=str, default="url_comparison.png")
    args = parser.parse_args()

    optimize_url(runs=args.runs, budget=args.budget, pop_size=args.pop_size, plot_file=args.plot)
