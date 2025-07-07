# Nora remote segmentation API

Repository for a FastAPI-based server for 3d image segmentation on [Nora](https://www.nora-imaging.com/).

## Requirements

```bash
uv sync
```

## Running the server
```bash
uv run python -m main
```

The server will be available at the adress specified in the --port argument, defaulting to 1527. 


## Running performance tests

Run predictions on a set of images contained in config["DATA_DIR"] (npz format) and compute the performance metrics (DSC, NSD, running time) for different prompt types.

```bash
uv run python -m tests.test_performance.py 
```

