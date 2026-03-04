import argparse
import csv
from dataclasses import dataclass
from html.parser import HTMLParser

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
from poly_sbst.crossover.html_test_suite_crossover import HTMLTestSuiteCrossover
from poly_sbst.generators.html_test_suite_generator import HTMLTestSuiteGenerator
from poly_sbst.mutation.html_test_suite_mutation import HTMLTestSuiteMutation
from poly_sbst.problems.html_test_suite_problem import HTMLTestSuiteProblem
from poly_sbst.sampling.abstract_sampling import AbstractSampling


@dataclass
class RunSummary:
    label: str
    best_ratio: float
    best_suite_size: int
    trace: np.ndarray


def build_html_grammar() -> AbstractGrammar:
    grammar = {
        "<start>": ["<html>", "<text>", ""],
        "<html>": ["<node>", "<node><node>", "<node><text>", "<text><node>"],
        "<node>": [
            "<open><children><close>",
            "<open><close>",
            "<selfclose>",
            "<open>",
            "<close>",
        ],
        "<open>": [
            "DIV_OPEN",
            "DIV_CLASS_OPEN",
            "DIV_ID_OPEN",
            "P_OPEN",
            "P_CLASS_OPEN",
            "A_OPEN",
            "A_HREF_OPEN",
            "A_HTTP_OPEN",
            "SPAN_OPEN",
            "SPAN_STYLE_OPEN",
            "UL_OPEN",
            "UL_CLASS_OPEN",
            "LI_OPEN",
            "B_OPEN",
            "I_OPEN",
        ],
        "<close>": ["DIV_CLOSE", "P_CLOSE", "A_CLOSE", "SPAN_CLOSE", "UL_CLOSE", "LI_CLOSE", "B_CLOSE", "I_CLOSE"],
        "<selfclose>": ["BR_SELF", "IMG_SELF", "HR_SELF", "INPUT_SELF"],
        "<children>": ["", "<text>", "<node>", "<node><node>", "<text><node>"],
        "<text>": ["x", "hello", "AMP_TOKEN", "LT_TOKEN", "GT_TOKEN", "APOS_TOKEN", "QUOT_TOKEN", "123", " "],
    }
    return AbstractGrammar(grammar)


def render_html_tokens(s: str) -> str:
    replacements = {
        "DIV_OPEN": "<div>",
        "DIV_CLASS_OPEN": "<div class='c'>",
        "DIV_ID_OPEN": "<div id='x'>",
        "DIV_CLOSE": "</div>",
        "P_OPEN": "<p>",
        "P_CLASS_OPEN": "<p class='txt'>",
        "P_CLOSE": "</p>",
        "A_OPEN": "<a>",
        "A_HREF_OPEN": "<a href='x'>",
        "A_HTTP_OPEN": "<a href='http://example.com'>",
        "A_CLOSE": "</a>",
        "SPAN_OPEN": "<span>",
        "SPAN_STYLE_OPEN": "<span style='color:red'>",
        "SPAN_CLOSE": "</span>",
        "UL_OPEN": "<ul>",
        "UL_CLASS_OPEN": "<ul class='list'>",
        "UL_CLOSE": "</ul>",
        "LI_OPEN": "<li>",
        "LI_CLOSE": "</li>",
        "B_OPEN": "<b>",
        "B_CLOSE": "</b>",
        "I_OPEN": "<i>",
        "I_CLOSE": "</i>",
        "BR_SELF": "<br/>",
        "IMG_SELF": "<img src='x'/>",
        "HR_SELF": "<hr/>",
        "INPUT_SELF": "<input>",
        "AMP_TOKEN": "&amp;",
        "LT_TOKEN": "<",
        "GT_TOKEN": ">",
        "APOS_TOKEN": "'",
        "QUOT_TOKEN": '"',
    }

    out = s
    for token, replacement in replacements.items():
        out = out.replace(token, replacement)
    return out


def running_best_trace(problem: HTMLTestSuiteProblem) -> np.ndarray:
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
        mutation=HTMLTestSuiteMutation(generator=generator),
        crossover=HTMLTestSuiteCrossover(cross_rate=0.9, max_length=generator.max_length),
        selection=selection,
        eliminate_duplicates=False,
    )


def run_strategy(label: str, strategy: str, runs: int, budget: int, pop_size: int) -> list[RunSummary]:
    summaries = []

    for _ in range(runs):
        seed = get_random_seed()
        grammar = build_html_grammar()
        generator = HTMLTestSuiteGenerator(grammar)

        original_generate_single = generator.generate_single_test

        def html_generate_single_test():
            return render_html_tokens(original_generate_single())

        generator.generate_single_test = html_generate_single_test

        executor = AbstractExecutor(HTMLParser().feed)
        problem = HTMLTestSuiteProblem(executor)

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

    plt.title("HTMLParser.feed: meilleure valeur cumulative de nl/nt par évaluation")
    plt.xlabel("Évaluation")
    plt.ylabel("Meilleur nl/nt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary(results: dict[str, list[RunSummary]]):
    print("=== HTML Optimization Summary ===")
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


def optimize_html(runs: int = 5, budget: int = 5000, pop_size: int = 50, plot_file: str = "html_comparison.png"):
    results = {
        "GA + TournamentSelection": run_strategy("GA + TournamentSelection", "tournament", runs, budget, pop_size),
        "GA + RandomSelection": run_strategy("GA + RandomSelection", "random", runs, budget, pop_size),
        "RandomSearch": run_strategy("RandomSearch", "random_search", runs, budget, pop_size),
    }

    plot_comparison(results, plot_file)
    export_run_csv(results, "html_results_runs.csv")
    export_trace_csv(results, "html_results_trace.csv")
    print_summary(results)
    print(f"Plot saved to: {plot_file}")
    print("CSV saved to: html_results_runs.csv, html_results_trace.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize HTMLParser.feed test suites with GA/RandomSearch.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--plot", type=str, default="html_comparison.png")
    args = parser.parse_args()

    optimize_html(runs=args.runs, budget=args.budget, pop_size=args.pop_size, plot_file=args.plot)
