from poly_sbst.problems.abstract_problem import AbstractProblem
from poly_sbst.common.abstract_executor import AbstractExecutor


class HTMLTestSuiteProblem(AbstractProblem):
    def __init__(self, executor: AbstractExecutor, n_var: int = 1, n_obj=1, n_ieq_constr=0, xl=None, xu=None):
        super().__init__(executor, n_var, n_obj, n_ieq_constr, xl, xu)
        self.executor = executor
        self._name = "HTMLTestSuiteProblem"
        self.execution_data = {}
        self.n_evals = 0

    def _evaluate(self, x, out, *_args, **_kwargs):
        tests = x[0]
        self.executor._full_coverage = []
        self.executor._coverage = set()

        total_time = 0.0
        total_exceptions = 0

        for test in tests:
            exceptions, execution_time, _ = self.executor._execute_input(test)
            total_exceptions += exceptions
            total_time += execution_time

        nb_exec = max(1, len(tests))
        coverage_count = len(self.executor._coverage)
        ratio = coverage_count / nb_exec

        out["F"] = -ratio
        self.execution_data[self.n_evals] = {
            "input": tests,
            "coverage": coverage_count,
            "nb_exec": len(tests),
            "ratio": ratio,
            "execution_time": total_time,
            "exceptions": total_exceptions,
        }
        self.n_evals += 1
