# Standard library
import glob
from os import path
from comoving_rv.longslit import GlobImageFileCollection

def main(data_path):

    for night_path in glob.glob(path.join(data_path, 'n?')):
        ic = GlobImageFileCollection(night_path)

        for hdu, fname in ic.hdus(return_fname=True):
            n_saturated = (hdu.data > 65530).sum()
            print('{0}: IMAGETYP={1}, OBJECT={2}, {3} saturated pixels'
                  .format(fname, hdu.header['IMAGETYP'], hdu.header['OBJECT'],
                          n_saturated))

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    parser.add_argument('-p', '--path', dest='path', required=True,
                        help='Path containing night directories (n1, n2, ..)')

    args = parser.parse_args()

    main(data_path=args.path)
