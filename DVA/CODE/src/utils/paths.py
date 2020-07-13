import os

# Project directory (/DVA_Project)
ROOT = os.path.dirname(__file__).rsplit(os.path.sep, maxsplit=2)[0]

SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
OUTPUT = os.path.join(ROOT, "output")
DATA_OCT = os.path.join(DATA, "QR Data October 2019")
DATA_DERIVED = os.path.join(DATA, "derived data")
TRC_MA = os.path.join(DATA_DERIVED, "TRC with moving averages")
TRC_ML = os.path.join(DATA_DERIVED, "ML")

TRC = os.path.join(DATA, "TRC")
TRC_138 = os.path.join(DATA, "TRC_C138")
TRC_195 = os.path.join(DATA, "TRC_195")
MODELS = os.path.join(ROOT, 'models')

TRC_LIST = ['TRC_NORTHGATE-SANDGATE-201701311116_coords.csv',
            'TRC_NORTHGATE-SANDGATE-201707041300_coords.csv',
            'TRC_NORTHGATE-SANDGATE-201710171136_coords.csv',
            'TRC_NORTHGATE-SANDGATE-201801301141_coords.csv']
