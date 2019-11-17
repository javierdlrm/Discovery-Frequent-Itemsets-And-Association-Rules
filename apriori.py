import itertools


class Apriori:
    """ This class comprehends an implementation of Apriori algorithm.
        Ctor: - Items: Path to file with items in buckets to apply Apriori algorithm
    """

    # defaults
    support = 0.5
    confidence = 0.75
    duplicates = False
    sep = ' '

    baskets = None
    candidates = None
    frequent_itemsets = None
    association_rules = None

    def __init__(self, path, n=None, sep=None, duplicates=None, verbose=False):
        self.load_baskets(path, n=n, sep=sep, duplicates=duplicates, verbose=verbose)

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

        if verbose: print("Reading {} lines...".format(count))

    def get_singletons(self, verbose=False):
        """ Get singletons of the list of baskets. """

        singletons = set()
        for basket in self.baskets:
            for item in basket:
                singletons.add(frozenset([item]))
        return singletons

    def compute(self, support=None, verbose=False):
        """ Computes apriori algorithm with a specified support
            Arguments:  - Support: support of the frequent itemsets
        """

        if support: self.support = support

        k = 0

        # C1
        self.candidates = [self.get_singletons(verbose=verbose)]
        if verbose: print("C{}: {}".format(k + 1, self.candidates[k]))

        # L1
        self.frequent_itemsets = [self.get_frequent_itemsets(self.candidates[k], support, verbose=verbose)]
        if verbose: print("L{}: {}".format(k + 1, self.frequent_itemsets[k]))

        # Ck, Lk
        while (len(self.frequent_itemsets[k]) > 0):
            k += 1
            new_candidates = self.get_candidates([x for x in self.frequent_itemsets[k - 1].keys()], verbose=verbose)
            if not new_candidates: break  # stop if not more candidates

            new_frequent_itemsets = self.get_frequent_itemsets(new_candidates, support, verbose=verbose)
            if not new_frequent_itemsets: break  # stop if not more frequent itemsets

            self.candidates.append(new_candidates)
            self.frequent_itemsets.append(new_frequent_itemsets)

            if verbose:
                print("C{}: {}".format(k + 1, self.candidates[k]))
                print("L{}: {}".format(k + 1, self.frequent_itemsets[k]))

        return self.candidates, self.frequent_itemsets

    def get_candidates(self, frequent_itemsets, verbose=False):
        """ Get new candidates from frequent itemsets.
            Arguments:  - Frequent itemsets: Lk-1, frequent itemsets of the previous iteration.
        """

        if not frequent_itemsets: return []

        candidates = []
        len_fi = len(frequent_itemsets)
        k = len(frequent_itemsets[0]) + 1

        for i in range(len_fi):
            targets = {}
            for j in range(i + 1, len_fi):
                intersect = frequent_itemsets[i] & frequent_itemsets[j]
                if len(intersect) == k - 2:
                    target = frequent_itemsets[j] - intersect
                    if target not in targets:
                        targets[target] = 0
                    targets[target] = targets[target] + 1
            for key, value in targets.items():
                if value == k - 1:
                    candidates.append(key | frequent_itemsets[i])
        return candidates

    def get_frequent_itemsets(self, candidates, support=None, verbose=False):
        """ Get frequent itemsets from candidates.
            Arguments:  - Candidates: candidates to take frequent itemsets from, over specified support.
                        - Support: support 
        """

        if support: self.support = support

        frequent_itemsets = {}
        for candidate in candidates:
            count = self.get_support(candidate)
            if count >= support:
                frequent_itemsets[candidate] = count
        return frequent_itemsets

    def get_association_rules(self, confidence=None, verbose=False):
        """ Get association rules from frequent itemsets with a specified confidence.
            Arguments:  - Frequent itemsets: frequent itemsets to extract association rules from.
                        - Baskets: all the baskets, original dataset
                        - Confidence: minimum confidence of the rules
        """

        if confidence: self.confidence = confidence
        if not self.frequent_itemsets: self.compute(verbose=verbose)

        association_rules = {}
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
                    if confidence >= self.confidence:
                        association_rules[rule] = (confidence, support)
                    else:
                        mistrusts.append(consequent)

        return association_rules

    def get_support(self, subset):
        assert self.baskets
        return sum(subset.issubset(basket) for basket in self.baskets)

    def get_confidence_and_support(self, rule):
        assert self.baskets
        union_support = self.get_support(rule[0] | rule[1])
        return round(float(union_support / self.get_support(rule[0])), 2), union_support
