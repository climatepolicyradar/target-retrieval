target-retrieval
==============================

Retrieving targets from climate laws and policies

## Quickstart

1. create a virtual environment
2. `make install` - install dependencies and pre-commit hooks
3. `make sync_data_from_s3` - get data from s3

## Updating data after Google Sheets changes

The command `make data` checks that the data is valid (for target retrieval only - not all the columns) and creates a CSV in the `data/interim` folder.