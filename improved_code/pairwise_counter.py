from typing import Any, Dict, Optional, NamedTuple
from scipy import sparse
import numba
import numpy as np

# need for log(0) where pair_count = 0
# do not affect results
EPS = 1e-100


class Stats(NamedTuple):
    pair_count: float
    count_1: float
    count_2: float
    total: float


class PairwiseCounter:
    def __init__(
        self,
        counts_matrix: sparse.csr_matrix,
        index_mapper: Dict[Any, int],
        total_key: Any,
    ):
        """
        Class for calculating some pair statistics.
        :param counts_matrix: sparse matrix of pairs
        :param index_mapper: dict from key to index in matrix
        :param total_key: key to count size of the data by line
        (total_key, total_key, value)
        """
        self.counts_matrix = counts_matrix
        self.index_mapper = index_mapper
        self.total_key = total_key
        total_index = index_mapper[total_key]
        self.total = self.counts_matrix[total_index, total_index]

    @staticmethod
    @numba.jit(nopython=True)
    def _get_csr_matrix_value(
        data: np.ndarray, indptr: np.ndarray, indices: np.ndarray, i: int, j: int
    ) -> Optional[int]:
        for row in range(len(indptr) - 1):
            for col in range(indptr[row], indptr[row + 1]):
                if row == i and indices[col] == j:
                    return data[col]

    def get_stats(self, key_1: Any, key_2: Any) -> Optional[Stats]:
        index_1 = self.index_mapper.get(key_1)
        index_2 = self.index_mapper.get(key_2)

        if index_1 is None or index_2 is None:
            return None

        pair_count = self._get_csr_matrix_value(
            self.counts_matrix.data,
            self.counts_matrix.indptr,
            self.counts_matrix.indices,
            index_1,
            index_2,
        )
        count_1 = self._get_csr_matrix_value(
            self.counts_matrix.data,
            self.counts_matrix.indptr,
            self.counts_matrix.indices,
            index_1,
            index_1,
        )
        count_2 = self._get_csr_matrix_value(
            self.counts_matrix.data,
            self.counts_matrix.indptr,
            self.counts_matrix.indices,
            index_2,
            index_2,
        )

        if pair_count is None:
            pair_count = 0

        if not count_1 or not count_2:
            return None

        return Stats(
            pair_count=float(pair_count),
            count_1=float(count_1),
            count_2=float(count_2),
            total=float(self.total),
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _count_pmi(
        pair_count: float, total: float, count_1: float, count_2: float
    ) -> Optional[float]:
        return (
            np.log(pair_count + EPS) + np.log(total) - np.log(count_1) - np.log(count_2)
        )

    def calculate_pmi(self, key_1: Any, key_2: Any) -> Optional[float]:
        """
        Calculates by formula: PMI
        PMI = log(p(x,y)/(p(x)p(y)))
        :param key_1: key 1
        :param key_2: key 2
        :return: weighted PMI
        """

        stats = self.get_stats(key_1, key_2)
        if stats is None:
            return None

        return self._count_pmi(
            stats.pair_count, stats.total, stats.count_1, stats.count_2
        )

    def to_dict(self) -> Dict[str, Any]:
        counts_matrix_dict = dict(
            data=self.counts_matrix.data.tolist(),
            indices=self.counts_matrix.indices.tolist(),
            indptr=self.counts_matrix.indptr.tolist(),
            shape=self.counts_matrix.shape,
        )
        return dict(
            counts_matrix=counts_matrix_dict,
            index_mapper=self.index_mapper,
            total_key=self.total_key,
        )

    @staticmethod
    def from_dict(params_dict: Dict[str, Any]):
        counts_matrix = sparse.csr_matrix(
            (
                params_dict["counts_matrix"]["data"],
                params_dict["counts_matrix"]["indices"],
                params_dict["counts_matrix"]["indptr"],
            ),
            shape=params_dict["counts_matrix"]["shape"],
        )
        return PairwiseCounter(
            counts_matrix=counts_matrix,
            index_mapper=params_dict["index_mapper"],
            total_key=params_dict["total_key"],
        )