import pathlib
from ncmapss import NcmapssLoader


def main():
    print('Creating datasets...')
    DATA_DIR = pathlib.Path.cwd().parent / 'data'
    # Make a raw data directory
    RAW_DATA_DIR = DATA_DIR / 'raw'
    RAW_DATA_DIR.mkdir(exist_ok=True)
    # Make a processed data directory
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    DECIMATION = 5

    files = [x for x in DATA_DIR.iterdir() if x.is_file() and 'Validation' not in x.name]
    files.remove(DATA_DIR / 'N-CMAPSS_DS08d-010.h5')

    for file in files:
        print(f'Processing {file.name}')
        ncmapss = NcmapssLoader(DATA_DIR, file.name, decimation=DECIMATION)
        ncmapss.create_dataset()
        ncmapss.save_dataset()

    print('Done!')


if __name__ == '__main__':
    main()