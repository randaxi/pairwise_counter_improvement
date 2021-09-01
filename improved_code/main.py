import json
import os
import typing as tp
from joblib import Parallel, delayed
from math import ceil
from tqdm.auto import tqdm
import time
from pairwise_counter import PairwiseCounter

start_time = time.time()

with open("../data/product_pairwise_counter.txt", "r", encoding="utf8") as infile:
    pairwise_counter = PairwiseCounter.from_dict(json.load(infile))

product_ids = [
    product_id
    for product_id in pairwise_counter.index_mapper.keys()
    if product_id != pairwise_counter.total_key
]

CPU_COUNT = os.cpu_count()
MAX_TOP_CANDIDATES: int = 10
most_co_occurring_products: tp.Dict[str, tp.List[str]] = dict()


def splitting_list(data: tp.List[str], number_of_parts: int) -> tp.Generator[tp.List[str]]:
    part_len = ceil(len(data) / number_of_parts)
    for k in range(number_of_parts):
        yield data[part_len * k: part_len * (k + 1)]


def main(part_lst: tp.List[str]) -> tp.Dict[str, tp.List[str]]:
    products = dict()
    for key_1 in tqdm(part_lst, desc="outer loop"):
        candidates: tp.List[tp.Tuple[str, float]] = []
        for key_2 in product_ids:
            if key_1 == key_2:
                continue

            pmi = pairwise_counter.calculate_pmi(key_1, key_2)
            if pmi is None:
                continue

            candidates.append((key_2, pmi))

        top_candidates = sorted(candidates, key=lambda p: p[1], reverse=True)[
            :MAX_TOP_CANDIDATES
        ]

        products[key_1] = [product_id for product_id, pmi in top_candidates]
    return products


list_of_dictionaries = Parallel(n_jobs=-1)(
    delayed(main)(lst) for lst in splitting_list(product_ids, CPU_COUNT)
)

for dictionaries in list_of_dictionaries:
    for key, value in dictionaries.items():
        most_co_occurring_products[key] = value

with open("../data/most_co_occurring_products_numba_joblib.txt", "w") as outfile:
    json.dump(most_co_occurring_products, outfile)

print(f"program running time: {time.time() - start_time} seconds")
