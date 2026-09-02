class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.max_features = self.max_features or len(X[0])
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = len(X), len(X[0])

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1:
            return Node(value=max(set(y), key=y.count))

        feat_idxs = random.sample(
            range(n_feats),
            min(n_feats, self.max_features)
        )

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return Node(value=max(set(y), key=y.count))

        left_idxs = [
            i for i, row in enumerate(X)
            if row[best_feat] <= best_thresh
        ]

        right_idxs = [
            i for i, row in enumerate(X)
            if row[best_feat] > best_thresh
        ]

        left = self._build_tree(
            [X[i] for i in left_idxs],
            [y[i] for i in left_idxs],
            depth + 1
        )

        right = self._build_tree(
            [X[i] for i in right_idxs],
            [y[i] for i in right_idxs],
            depth + 1
        )

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left,
            right=right
        )

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            thresholds = set(row[feat_idx] for row in X)

            for thr in thresholds:
                gain = self._information_gain(X, y, feat_idx, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, X, y, feat_idx, thr):
        parent_entropy = self._entropy(y)

        left_y = [
            y[i] for i, row in enumerate(X)
            if row[feat_idx] <= thr
        ]

        right_y = [
            y[i] for i, row in enumerate(X)
            if row[feat_idx] > thr
        ]

        if not left_y or not right_y:
            return 0

        n = len(y)
        n_l = len(left_y)
        n_r = len(right_y)

        e_l = self._entropy(left_y)
        e_r = self._entropy(right_y)

        return parent_entropy - (
            (n_l / n) * e_l +
            (n_r / n) * e_r
        )

    def _entropy(self, y):
        hist = [y.count(c) for c in set(y)]

        return -sum(
            (c / len(y)) * math.log2(c / len(y))
            for c in hist if c > 0
        )

    def predict(self, X):
        return [
            self._traverse_tree(x, self.root)
            for x in X
        ]

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10,
                 min_samples_split=2, max_features=None):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )

            # Bootstrap
            indices = [
                random.randint(0, len(X) - 1)
                for _ in range(len(X))
            ]

            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = [
            tree.predict(X)
            for tree in self.trees
        ]

        sample_preds = zip(*tree_preds)

        return [
            max(set(sample), key=sample.count)
            for sample in sample_preds
        ]