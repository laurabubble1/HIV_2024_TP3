from poly_sbst.generators.abstract_generator import AbstractGenerator
from poly_sbst.common.abstract_grammar import AbstractGrammar
import numpy as np
import random


class HTMLTestSuiteGenerator(AbstractGenerator):
    """Generates HTML parser test suites from a grammar."""

    def __init__(self, grammar: AbstractGrammar) -> None:
        super().__init__()
        self._name = "HTMLTestSuiteGenerator"
        self.grammar = grammar
        self.max_length = 40
        self.min_length = 1

    @property
    def name(self) -> str:
        return self._name

    def cmp_func(self, x: str, y: str) -> float:
        return 0.0

    def generate_single_test(self) -> str:
        return self.grammar.generate_input()

    def generate_random_test(self):
        n_tests = random.randint(self.min_length, self.max_length)
        return np.array([self.generate_single_test() for _ in range(n_tests)], dtype=object)
