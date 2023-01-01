# LogTransform

**Description**: The following is a GenePattern module written in Python 3. It processes a GCT file by taking the natural log of all positive data-points (turning all other values to 0), and outputs the new resulting GCT file.

**Author**: Omar Halawa, Mesirov Lab - University of California, San Diego

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Summary

This repository is a GenePattern module written in [Python 3](https://www.python.org/download/releases/3.0/).

This module takes in an input GCT file, process it by taking the natural log of each of its positive data points (turning all non-positive values into 0), and outputs a new GCT file of the processed data.


## Source Links
* [The GenePattern LogTransform source repository](https://github.com/omarhalawa3301/log_normalize)
* LogTransform uses the [genepattern/notebook-python39:latest Docker image](https://hub.docker.com/layers/genepattern/notebook-python39/21.08/images/sha256-12b175ff4472cfecef354ddea1d7811f2cbf0ae9f114ede11d789b74e08bbc03?context=explore)

## Usage
python log_normalize.py &lt;filename&gt; (-v)

## Parameters

| Name | Description | Default Value |
---------|--------------|----------------
| filename * |  The input file to be read in .gct format | No default value |
| verbose | Optional parameter to increase output verbosity | False |

\*  required

## Input Files

1. filename  
    This is the input file which will be read in by the python script and ultimately will be processed through log normalization of positive values. The parameter expects a GCT file (.gct extension).
    
## Output Files

1. result.gct\
    The log-normalized version of the input GCT file's data. Non-positive values all become 0.
2. stdout.txt\
    This is standard output from the Python script. Sometimes helpful for debugging.

## Example Data

Input:  
[all_aml_train.gct](https://github.com/omarhalawa3301/log_normalize/blob/main/data/all_aml_train.gct)

Output:  
[result.gct](https://github.com/omarhalawa3301/log_normalize/blob/main/data/result.gct)


## Requirements

Requires the [genepattern/notebook-python39:latest Docker image](https://hub.docker.com/layers/genepattern/notebook-python39/21.08/images/sha256-12b175ff4472cfecef354ddea1d7811f2cbf0ae9f114ede11d789b74e08bbc03?context=explore).

## License

`LogTransform` is distributed under a modified BSD license available [here](https://github.com/omarhalawa3301/log_normalize/blob/main/LICENSE.txt)

## Version Comments

| Version | Release Date | Description                                 |
----------|--------------|---------------------------------------------|
| 1.0.0 | Oct 4, 2022 | Initial version for team use. |
