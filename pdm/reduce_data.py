"""Resample and downcast NCMAPSS dataset.

Script which reads a raw PHMAP2021 data directory and
applies downcasting to numeric data types while resampling
to every nth observation
"""
import pathlib
from ncmapss import NcmapssLoader


def main():
    """Read raw dir and downcast/resample.

    Avoid 'N-CMAPSS_DS08d-010.h5' as its corrupted
    """
    print('Creating datasets...')
    DATA_DIR = pathlib.Path.cwd().parent / 'data'

    # Make a raw data directory
    RAW_DATA_DIR = DATA_DIR / 'raw'
    RAW_DATA_DIR.mkdir(exist_ok=True)

    # Make a processed data directory
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    DECIMATION = 5

    # Process all training files except Validation
    # Ignored DS08 since it's corrupted
    files = []
    for f in RAW_DATA_DIR.iterdir():
        if f.is_file() and 'Validation' not in f.name:
            files.append(f)

    files.remove(RAW_DATA_DIR / 'N-CMAPSS_DS08d-010.h5')

    for file in files:
        print(f'Processing {file.name}')
        ncmapss = NcmapssLoader(RAW_DATA_DIR, file.name, decimation=DECIMATION)
        ncmapss.create_dataset()
        ncmapss.save_dataset()

    print('Done!')


if __name__ == '__main__':
    main()
