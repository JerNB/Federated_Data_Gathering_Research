# Federated_Data_Gathering_Research

Research code over the MovieLens `ml-latest` dataset (snapshot generated
2023-07-20: 33,832,162 ratings and 2,328,315 tag applications across 86,537
movies from 330,975 users).

## Data setup

The raw dataset is **not stored in this repository**. `ratings.csv` (890 MB) and
`genome-scores.csv` (497 MB) exceed GitHub's hard limit of 100 MiB per file, and
a 1.5 GB working tree would make every clone expensive.

```sh
scripts/fetch_data.sh            # download + verify into ./ml-latest
scripts/fetch_data.sh --verify   # re-verify an existing ./ml-latest
```

The script downloads a single 356 MB archive from this repository's
[`data-2023-07-20` release](https://github.com/JerNB/Federated_Data_Gathering_Research/releases/tag/data-2023-07-20)
and checks every extracted file against `data/ml-latest.sha256`, so a truncated
or substituted download fails loudly instead of silently skewing results.

Expected layout after fetching:

| file | size |
| --- | --- |
| `ml-latest/ratings.csv` | 890 MB |
| `ml-latest/genome-scores.csv` | 497 MB |
| `ml-latest/tags.csv` | 81 MB |
| `ml-latest/movies.csv` | 4.0 MB |
| `ml-latest/links.csv` | 1.8 MB |
| `ml-latest/genome-tags.csv` | 18 KB |
| `ml-latest/README.txt` | 9.6 KB |

## Dataset license and citation

The MovieLens license permits redistribution, including transformations, as long
as it is distributed under the same conditions; it also prohibits commercial use
without permission from GroupLens. Full terms are in `ml-latest/README.txt`
after fetching. Neither the University of Minnesota nor GroupLens endorses this
work.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4:
> 19:1–19:19. <https://doi.org/10.1145/2827872>
