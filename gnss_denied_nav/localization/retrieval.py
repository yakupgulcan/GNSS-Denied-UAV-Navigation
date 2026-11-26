"""Candidate retrieval: which map frames is the query worth matching against?

* Spatial prior (MapDatabase.frames_near): the normal case once localized.
* VotingRetriever: no prior (first fix / relocalization). Every query
  descriptor votes for the map frame of its nearest database descriptor.
"""
import cv2
import numpy as np


class VotingRetriever:
    """Global descriptor voting over the whole map.

    Neighbouring map frames overlap, so a ground feature usually has correct
    nearest neighbours in several frames. A plain Lowe ratio test (1st vs 2nd
    neighbour) would reject exactly those features. Instead the 1st neighbour
    is compared with the closest neighbour from a frame at least
    ``distinct_m`` away, i.e. one that cannot show the same ground.
    """

    def __init__(self, db, binary, ratio=0.8, k=6, distinct_m=60.0):
        self.ratio = ratio
        self.k = k
        self.distinct_m = distinct_m
        self.binary = binary
        self.enu = db.enu
        self.n_frames = len(db)
        self.frame_of = np.repeat(np.arange(len(db)), np.diff(db.offsets)).astype(np.int32)
        if binary:
            index = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # LSH
            search = dict(checks=30)
        else:
            index = dict(algorithm=1, trees=5)  # KD-tree
            search = dict(checks=50)
        desc = db.desc if binary else db.desc.astype(np.float32)
        self.matcher = cv2.FlannBasedMatcher(index, search)
        self.matcher.add([desc])
        self.matcher.train()

    def retrieve(self, query, max_count, min_votes=6, exclude=()):
        """Return [(frame_index, votes)] best first."""
        if len(query) < 2:
            return []
        excluded = set(int(i) for i in exclude)
        votes = np.zeros(self.n_frames, dtype=np.int32)
        # The index holds float32; learned extractors return float16 descriptors.
        desc = query.desc if self.binary else np.asarray(query.desc, np.float32)
        for neighbours in self.matcher.knnMatch(desc, k=self.k):
            best = None
            for m in neighbours:
                f = self.frame_of[m.trainIdx]
                if f in excluded:
                    continue
                if best is None:
                    best = (f, m.distance)
                    continue
                if np.hypot(*(self.enu[f] - self.enu[best[0]])) >= self.distinct_m:
                    if best[1] >= self.ratio * m.distance:
                        best = None  # ambiguous between two different places
                    break
            if best is not None:
                votes[best[0]] += 1
        order = np.argsort(votes)[::-1][:max_count]
        return [(int(i), int(votes[i])) for i in order if votes[i] >= min_votes]
