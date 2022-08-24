import torch
calcdevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import models.addons as addons
# import models.catboost as catboost
import models.functions as functions
from models.models import Model,Model3
from models.oanet4s import DLOA,Model0
from models.apply import apply,valid_bs,applymodels
from models.inference import inference, test, getests
from models.loadmodels import loadmodels


