"""Phase 7 chronological leave-latest-positive-out evaluation protocol for MovieLens 1M."""

MIN_TRAIN_HISTORY = 5
RELEVANT_THRESHOLD = 4.0


def split_outer_temporal(ratings, threshold=RELEVANT_THRESHOLD, min_train_history=MIN_TRAIN_HISTORY):
    """Split each user chronologically; latest rating >= threshold becomes the single test target.

    Users with fewer than min_train_history strictly-earlier rows before their latest positive
    rating are excluded entirely (documented eligibility rule applied identically to every model).
    Rows with identical timestamps are ordered by movie_id for determinism.
    """
    grouped = {}
    for row in ratings:
        grouped.setdefault(row[0], []).append(row)
    train, test = [], []
    excluded = []
    for user_id in sorted(grouped):
        rows = sorted(grouped[user_id], key=lambda row: (row[3], row[1]))
        pos_indices = [i for i, row in enumerate(rows) if row[2] >= threshold]
        if not pos_indices:
            excluded.append(user_id)
            continue
        last_pos = pos_indices[-1]
        if last_pos < min_train_history:
            excluded.append(user_id)
            continue
        train.extend(rows[:last_pos])
        test.append(rows[last_pos])
    return train, test, excluded


def split_inner_chronological(ratings, fraction=0.2):
    """Per-user chronological inner train/validation split of outer_train (earliest 80% / latest 20%)."""
    if not 0 < fraction < 1:
        raise ValueError("inner validation fraction must be between 0 and 1")
    grouped = {}
    for row in ratings:
        grouped.setdefault(row[0], []).append(row)
    train, validation = [], []
    for rows in grouped.values():
        rows.sort(key=lambda row: (row[3], row[1]))
        count = min(len(rows) - 1, max(1, int(len(rows) * fraction))) if len(rows) > 1 else 0
        train.extend(rows[:-count] if count else rows)
        if count:
            validation.extend(rows[-count:])
    return train, validation


def group_relevant_by_user(rows, threshold=RELEVANT_THRESHOLD):
    """Relevant movie sets for evaluation; threshold >= 4.0 matches project convention."""
    relevant = {}
    for user_id, movie_id, rating, _ in rows:
        if rating >= threshold:
            relevant.setdefault(user_id, set()).add(movie_id)
    return relevant
