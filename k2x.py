import argparse
import os

from pykmp import KMP, read_kmp


def add_ext_if_missing(filename: str, ext: str):
    if not filename.endswith(ext):
        filename += ext
    return filename


def main(args):
    if args.type == 'k2x':
        args.input = add_ext_if_missing(args.input, '.kmp')
        if args.output is None:
            args.output = os.path.splitext(args.input)[0] + '.xlsx'
        else:
            args.output = add_ext_if_missing(args.output, '.xlsx')
        print(f'Converting {args.input} to {args.output}')
        read_kmp(args.input).to_excel(args.output)
    elif args.type == 'x2k':
        args.input = add_ext_if_missing(args.input, '.xlsx')
        if args.output is None:
            args.output = os.path.splitext(args.input)[0] + '.kmp'
        else:
            args.output = add_ext_if_missing(args.output, '.kmp')
        print(f'Converting {args.input} to {args.output}')
        KMP.from_excel(args.input).write(args.output)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "k2x converter",
        description=(
            "Converts between KMP and Excel (.xlsx) files. "
            "\nFor more details, please see Wiki page."
        )
    )
    parser.add_argument(
        'type', choices=['k2x', 'x2k'],
        help='Type of conversion. k2x: KMP to Excel, x2k: Excel to KMP.'
    )
    parser.add_argument('--input', '-i', help='Input file', required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output path. If not specified, the output file will be saved in the same directory as the input file.',
        default=None)

    args = parser.parse_args()
    main(args)
