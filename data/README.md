# Data

Raw datasets are downloaded locally and are not committed to Git. Source archives and extracted files are never rewritten by validation.

## MovieLens

- Provider: GroupLens Research, University of Minnesota.
- MovieLens 100K: <https://files.grouplens.org/datasets/movielens/ml-100k.zip>
- MovieLens 1M: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
- Official dataset page and terms: <https://grouplens.org/datasets/movielens/>

The downloader supports only `ml-100k` and `ml-1m`. MovieLens 25M is intentionally unsupported in Phase 1. Review GroupLens terms before redistributing any downloaded data. Keep original files intact; generated summaries belong outside `data/raw/`.

Expected local layout:

```text
data/raw/ml-100k/u.data
data/raw/ml-100k/u.item
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

The validator consumes ratings and movie metadata as read-only streams. It uses a temporary SQLite index for exact duplicate-interaction detection without retaining all interactions in Python memory; temporary files are removed after validation.
