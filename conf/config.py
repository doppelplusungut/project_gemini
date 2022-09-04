from easydict import EasyDict

CONF = EasyDict()

#Training
CONF.TRAIN = EasyDict()

CONF.TRAIN.MANUAL_SEED = 42
CONF.TRAIN.EPOCHS = 2
CONF.TRAIN.BATCH_SIZE = 16
CONF.TRAIN.NUM_WORKERS = 12

#Validation
CONF.VAL = EasyDict()
CONF.VAL.BATCH_SIZE = 16
CONF.VAL.NUM_WORKERS = 12

#Model
CONF.MODEL = EasyDict()
CONF.MODEL.N_OUT = 62