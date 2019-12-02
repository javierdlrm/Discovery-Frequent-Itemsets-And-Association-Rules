import itertools
import time
from tqdm.auto import tqdm


class Apriori:
    """ This class comprehends an implementation of Apriori algorithm.
        Ctor: - Items: Path to file with items in buckets to apply Apriori algorithm
    """

    # defaults
    min_support = 0.01  # 1%
    min_confidence = 0.75
    duplicates = False
    sep = ' '

    baskets = None
    n_baskets = 0
    candidates = None
    last_candidates_basket_indexes = None
    frequent_itemsets = None
    association_rules = None

    def __init__(self, path=None, n=None, sep=None, duplicates=None, verbose=False):
        if path: self.load_baskets(path, n=n, sep=sep, duplicates=duplicates, verbose=verbose)

    def load_baskets(self, path, n=None, sep=None, duplicates=None, verbose=False):
        """ Load baskets from file.
            Arguments:  - Path: file path
                        - Sep: separator of items in same basket
                        - Duplicates: whether there are duplicates in the baskets
        """

        if duplicates: self.duplicates = duplicates
        if sep: self.sep = sep

        self.baskets = []
        count = 0
        with open(path) as f:
            for line in f:
                items = line.rstrip().split(self.sep)

                # remove duplicates in basket (save memory)
                if self.duplicates:
                    items = frozenset(items)

                self.baskets.append(items)

                count += 1
                if count >= n:
                    break
        
        # update number of baskets
        self.n_baskets = count

        if verbose: print("Reading {} lines...".format(count))

    def get_singletons(self, verbose=False):
        """ Get singletons of the list of baskets. """

        self.last_candidates_basket_indexes = {}

        singletons = set()
        for i, basket in enumerate(self.baskets):
            for item in basket:
                itemset = frozenset([item])

                # add basket index
                if itemset not in self.last_candidates_basket_indexes:
                    self.last_candidates_basket_indexes[itemset] = set([i])
                else:
                    self.last_candidates_basket_indexes[itemset].add(i)

                # add itemset
                singletons.add(itemset)

        return singletons

    def compute(self, min_support=None, stop=True, verbose=False):
        """ Computes apriori algorithm with a specified support
            Arguments:  - Min_support: minimum support for the frequent itemsets
        """
        if min_support: self.min_support = min_support

        k = 0

        print("# Computing C1 and L1...")
        start = time.time()

        # C1
        start_i = time.time()
        self.candidates = [self.get_singletons(verbose=verbose)]
        print("- {} candidates in {} ms".format(len(self.candidates[k]), (time.time() - start_i) * 1000))
        if verbose: print("C{}: {}".format(k + 1, self.candidates[k]))

        # L1
        start_i = time.time()
        self.frequent_itemsets = [self.get_frequent_itemsets(self.candidates[k], self.min_support, stop=stop, verbose=verbose)]
        print("- {} frequent itemsets in {} ms".format(len(self.frequent_itemsets[k]), (time.time() - start_i) * 1000))
        if verbose: print("L{}: {}".format(k + 1, self.frequent_itemsets[k]))

        print("- {} iteration completed in {} ms".format(k + 1, (time.time() - start) * 1000))

        while (len(self.frequent_itemsets[k]) > 0):
            k += 1

            print("# Computing C{} and L{}...".format(k + 1, k + 1))
            start = time.time()

            # Ck
            start_i = time.time()
            new_candidates = self.get_candidates([x for x in self.frequent_itemsets[k - 1].keys()], verbose=verbose)

            if not new_candidates: break  # stop if not more candidates
            print("- {} candidates in {} ms".format(len(new_candidates), (time.time() - start_i) * 1000))

            # Lk
            start_i = time.time()
            new_frequent_itemsets = self.get_frequent_itemsets(new_candidates, self.min_support, stop=stop, verbose=verbose)

            if not new_frequent_itemsets: break  # stop if not more frequent itemsets
            print("- {} frequent itemsets in {} ms".format(len(new_frequent_itemsets), (time.time() - start_i) * 1000))

            # save iteration
            self.candidates.append(new_candidates)
            self.frequent_itemsets.append(new_frequent_itemsets)

            if verbose:
                print("C{}: {}".format(k + 1, self.candidates[k]))
                print("L{}: {}".format(k + 1, self.frequent_itemsets[k]))

            print("- {} iteration completed in {} ms".format(k + 1, (time.time() - start) * 1000))

        return self.candidates, self.frequent_itemsets

    def get_candidates(self, frequent_itemsets, verbose=False):
        """ Get new candidates from frequent itemsets.
            Arguments:  - Frequent itemsets: Lk-1, frequent itemsets of the previous iteration.
        """

        if not frequent_itemsets: return []

        candidates = []
        len_fi = len(frequent_itemsets)
        k = len(frequent_itemsets[0]) + 1
        for i in tqdm(range(len_fi)):
            targets = {}

            frequent_itemset_1 = frequent_itemsets[i]

            for j in range(i + 1, len_fi):
                frequent_itemset_2 = frequent_itemsets[j]
                intersect = frequent_itemset_1 & frequent_itemset_2

                if len(intersect) == k - 2:
                    frequent_itemset_2_basket_indexes = self.last_candidates_basket_indexes[frequent_itemset_2]  # frequent_itemset_2 basket indexes

                    # update target
                    target = frequent_itemset_2 - intersect
                    if target not in targets:
                        frequent_itemset_1_basket_indexes = self.last_candidates_basket_indexes[frequent_itemset_1]  # frequent_itemset_1 basket indexes
                        targets[target] = (1, frequent_itemset_1_basket_indexes & frequent_itemset_2_basket_indexes)
                    else:
                        (c, indexes) = targets[target]
                        targets[target] = (c + 1, indexes & frequent_itemset_2_basket_indexes)

            for target, info in targets.items():
                if info[0] == k - 1:  # save candidate
                    candidate_set = target | frequent_itemset_1
                    candidates.append(candidate_set)
                    self.last_candidates_basket_indexes[candidate_set] = info[1]

        return candidates

    def get_frequent_itemsets(self, candidates, min_support=None, stop=True, verbose=False):
        """ Get frequent itemsets from candidates.
            Arguments:  - Candidates: candidates to take frequent itemsets from, over specified support.
                        - Min_support: Minimum support 
        """

        if min_support: self.min_support = min_support

        frequent_itemsets = {}
        for candidate in tqdm(candidates):
            count = self.get_support(candidate, stop=stop)
            if count >= self.min_support:
                frequent_itemsets[candidate] = count

        return frequent_itemsets

    def get_association_rules(self, min_confidence=None, verbose=False):
        """ Get association rules from frequent itemsets with a specified confidence.
            Arguments:  - Frequent itemsets: frequent itemsets to extract association rules from.
                        - Baskets: all the baskets, original dataset
                        - Min confidence: minimum confidence of the rules
        """

        if min_confidence: self.min_confidence = min_confidence
        if not self.frequent_itemsets: self.compute(verbose=verbose)

        self.association_rules = {}
        flatten_itemsets = [item for sublist in self.frequent_itemsets[1:] for item in sublist]
        for idx in range(len(flatten_itemsets) - 1, -1, -1):
            itemset = flatten_itemsets[idx]
            mistrusts = []
            for size in range(len(itemset) - 1, 0, -1):
                subsets = list(map(frozenset, itertools.combinations(itemset, size)))
                for antecedent in subsets:
                    consequent = itemset ^ antecedent

                    # If K,L,M -> N is below confidence, so is K,L -> M,N
                    if any(c.issubset(consequent) for c in mistrusts):
                        continue

                    rule = (antecedent, consequent)
                    confidence, support = self.get_confidence_and_support(rule)
                    if confidence >= self.min_confidence:
                        self.association_rules[rule] = (confidence, support)
                    else:
                        mistrusts.append(consequent)

        return self.association_rules

    def get_support(self, subset, stop=True):
        return float(len(self.last_candidates_basket_indexes[subset]) / self.n_baskets)

    def get_confidence_and_support(self, rule):
        union_support = self.get_support(rule[0] | rule[1], stop=False)
        return float(union_support / self.get_support(rule[0], stop=False)), union_support
