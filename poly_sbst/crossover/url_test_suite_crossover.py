from poly_sbst.crossover.abstract_crossover import AbstractCrossover
import numpy as np


class URLTestSuiteCrossover(AbstractCrossover):
	def __init__(self, cross_rate: float = 0.9, max_length: int = 40):
		super().__init__(cross_rate)
		self.max_length = max_length

	def _do_crossover(self, problem, a, b) -> tuple:
		if len(a) == 0 or len(b) == 0:
			return a, b

		cut_a = np.random.randint(0, len(a) + 1)
		cut_b = np.random.randint(0, len(b) + 1)

		child_a = np.concatenate([a[:cut_a], b[cut_b:]])
		child_b = np.concatenate([b[:cut_b], a[cut_a:]])

		if len(child_a) == 0:
			child_a = np.array([a[np.random.randint(len(a))]], dtype=object)
		if len(child_b) == 0:
			child_b = np.array([b[np.random.randint(len(b))]], dtype=object)

		if len(child_a) > self.max_length:
			child_a = child_a[:self.max_length]
		if len(child_b) > self.max_length:
			child_b = child_b[:self.max_length]

		return child_a, child_b
