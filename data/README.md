# Data

The raw monitoring data are not included in this repository.

Download the prepared 5-minute BHE monitoring data from the dataset cited in the manuscript:

- DOI: `10.5281/zenodo.12724484`

Run the workflow by passing the downloaded directory, CSV file, or ZIP archive to:

```bash
python code/run_all.py --raw-source PATH_TO_DATA
```

`data/metadata/bhe_metadata.csv` contains the BHE identifiers, vault allocation, coordinates, and horizontal pipe lengths used by the analysis.
