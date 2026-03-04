from poly_sbst.mutation.abstract_mutation import AbstractMutation
import numpy as np


class URLTestSuiteMutation(AbstractMutation):
	def __init__(self, mut_rate: float = 0.4, generator=None):
		super().__init__(mut_rate, generator)

	def _do_mutation(self, x) -> np.ndarray:
		possible_mutations = [
			self._delete_random_element,
			self._insert_random_element,
			self._replace_random_element,
		]
		mutator = np.random.choice(possible_mutations)
		return mutator(x)

	def _delete_random_element(self, suite):
		if len(suite) > self.generator.min_length:
			index_to_remove = np.random.randint(len(suite))
			suite = np.delete(suite, index_to_remove)
		return suite

	def _insert_random_element(self, suite):
		if len(suite) < self.generator.max_length:
			new_element = self.generator.generate_single_test()
			suite = np.append(suite, new_element)
		return suite

	def _replace_random_element(self, suite):
		if len(suite) == 0:
			return np.array([self.generator.generate_single_test()], dtype=object)

		index_to_replace = np.random.randint(len(suite))
		suite[index_to_replace] = self.generator.generate_single_test()
		return suite
