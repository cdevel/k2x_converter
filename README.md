# k2x
Converts between KMP and Excel (.xlsx) files. For more details, please see [Wiki page](https://wiki.tockdom.com/wiki/K2X_Converter).

# Dependencies
- **Python 3.10 or later**
- [numpy](https://numpy.org/)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
- [pandas](https://pandas.pydata.org/)
- [typing-extensions](https://pypi.org/project/typing-extensions/)

# Usage
```bash
python -m {k2x, x2k} -i <input> [-o <output>]
```

- `{k2x, x2k}`: Convert KMP to Excel (`k2x`) or Excel to KMP (`x2k`)
- `<input>`: Input file path
- `<output>`: Output file path (optional) if not specified, the output file will be saved in the same directory as the input file with the same name and the extension changed.

## Examples
### Convert KMP to Excel
```bash
python -m k2x -i <input.kmp> [-o <output.xlsx>]
```

### Convert Excel to KMP
```bash
python -m k2x -i <input.xlsx> [-o <output.kmp>]
```

# NOTE
- This application will be merged into [pykmp](https://github.com/cdevel/pykmp) in the future. Therefore, there are no plans to fix it unless it is a critical bug.
- `x2k` supports `.xlsx` that generated by older version (v0.1)
