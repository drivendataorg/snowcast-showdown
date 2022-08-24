getdays = lambda keys: list(sorted([tday for tday in keys if tday[:4].isnumeric() and len(tday) == 10]))

from features.constfeatures import getconstfeatures
from features.modisfeatures import getmodisfeatures
from features.datadict import getdatadict
from features.embeddings import getembindex
from features.inputs import getinputs

