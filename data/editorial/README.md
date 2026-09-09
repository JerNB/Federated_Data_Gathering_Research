# Editable data

This directory contains small tables that collaborators may edit through normal Git branches and pull requests.

Current editable tables:

```text
movies.csv
links.csv
```

They start as copies of the pinned MovieLens source tables. Add a new commit for every reviewed data change. Keep the large interaction and genome tables under `data/raw/` unchanged; they are the byte-preserving source mirror.

When oracle cluster assignments or other curated labels are created, add them here with a clear table name and document their source and commit in the experiment record.
