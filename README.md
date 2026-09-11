# mt-io

[![PyPI Version](https://img.shields.io/pypi/v/mt-io.svg)](https://pypi.python.org/pypi/mt-io)

**Version**: 0.0.5

This package is meant to be a repository for readers for various magnetotelluric data file types from various data loggers including:

| Data Logger | File Type | Reader | Maturity |
|-------------|-----------|--------|----------|
| LEMI 424 | txt | Y | medium |
| LEMI 423 | B423 | Y | medium |
| LEMI 417 | ? | In progress | weak | 
| Metronix | atss| Y | medium |
| Metronix | ats | In progress | weak |
| MiniSEED | mseed | Y | strong |
| NIMS | bin | Y | strong |
| Phoenix MTU-5C | td_*, JSON, bin | Y | medium |
| Phoenix MTU-5A | tbl* | Y | medium |
| UoA PR6-24 | BX, BY, BZ, EX, EY | Y | medium |
| UoA Orange | binary | Y | medium |
| USGS ASCII | ascii | Y | medium |
| ZEN | z3d | Y | strong |

This package will read the data into either a `mt_timeseries.ChannelTS` or a `mt_timeseries.RunTS` object depending on the data included in the file.

## Newer Readers

Recent additions include newer LEMI and University of Adelaide readers:

- `mt_io.lemi.read_lemi423` for LEMI-423 `B423` files
- `mt_io.lemi.read_lemi424` for LEMI-424 ASCII text files
- `mt_io.uoa.read_uoa` for PR6-24 Earth Data Logger channel files
- `mt_io.uoa.read_orange` for legacy Orange Box binary files

These readers return `mt_timeseries.RunTS` objects and are covered by the current test suite.

## Installation

### From Source

```bash
git clone https://github.com/kujaku11/mt-io.git
cd mt-io
pip install .
```

For development installation with editable mode:

```bash
pip install -e .
```

**Note**: This package is currently in development. PyPI and conda-forge releases will be available in future versions.

## Issues

If you encounter any problems or have suggestions for improvements, please report issues on the [GitHub Issues page](https://github.com/kujaku11/mt-io/issues).

When reporting an issue, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your Python version and operating system
- Relevant code snippets or error messages

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub
2. Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`)
3. Make your changes and commit them with clear, descriptive messages
4. Add tests for any new functionality
5. Ensure all tests pass
6. Push your branch to your fork (`git push origin feature/your-feature-name`)
7. Submit a pull request to the main repository

Please ensure your code follows the project's coding standards and includes appropriate documentation.

For major changes, please open an issue first to discuss what you would like to change.

## Test Data

If you have a small data set contribute to the repository [here](https://github.com/kujaku11/mth5_test_data).  





