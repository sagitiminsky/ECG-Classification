import wandb
import numpy as np
from coding.learning.data import DataModule
from coding.learning.model import PaperNet,ImagePredictionLogger
import torch
import pickle
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

class DatasetReport():
    def __init__(self,data_map):
        self.data_map=data_map

    def get_len(self,category):
        return len(category['A'])+len(category['B'])

    def category_report(self,state,gender,age_group):
        gender_code=0 if gender=='male' else 1
        age_code='<' if age_group else '>='
        print("state:{} gender:{} age_group:{} len:{}".format(state,gender,age_code,self.get_len(self.data_map[state][gender_code][age_code])))

    def full_report(self):
        for state in ['train','test']:
            for gender in ['male','female']:
                for age_group in [True,False]:
                    self.category_report(state=state,gender=gender,age_group=age_group)


if __name__=="__main__":

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    with open('data_map.pickle', 'rb') as handle:
        data_map = pickle.load(handle)

    dataset_report=DatasetReport(data_map)
    dataset_report.full_report()

    # Init our data pipeline
    data_map_url = "data_map"
    data_url = "STFT"
    gender = "male"
    under_50 = True
    is_train = True
    state = 'train'

    batch_size = 32
    input_shape = (1, 256, 256)

    super_classes = np.array(["CD", "HYP", "MI", "NORM", "STTC"])
    dm = DataModule(batch_size, data_map_url, data_url, gender, under_50, is_train)
    dm.prepare_data()
    dm.setup()

    with open('labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    label_hist = list(np.unique(labels, return_counts=True))
    label_hist[1] = label_hist[1] / sum(label_hist[1])
    plt.hist(labels)


    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    # Initialize wandb
    wandb.init(project='ECG_spec_classify')
    wandb_logger = WandbLogger(project='ECG_spec_classify', job_type='train')


    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=6,
       verbose=False,
       mode='min'
    )

    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=MODEL_CKPT,
        save_top_k=3,
        mode='min')

    # Init our model
    weight_decay=0.01
    #model = Net1(input_shape, len(super_classes), device,  weight_decay=weight_decay)
    model = PaperNet(input_shape, len(super_classes), device, weight_decay)

    for i in range(5):
        trainer = pl.Trainer(
            logger=wandb_logger,    # W&B integration
            log_every_n_steps=5,   # set the logging frequency
            gpus=-1,                # use all GPUs
            max_epochs=25,           # number of epochs
            #deterministic=True,     # keep it deterministic
            auto_lr_find=True,
            callbacks=[
                       ImagePredictionLogger(val_samples, super_classes),
                       ModelCheckpoint(monitor='val_loss', filename=MODEL_CKPT, save_top_k=3, mode='min')
                      #  EarlyStopping(monitor='val_loss',patience=3,verbose=False,mode='min')
                        ] # see Callbacks section
        )


        trainer.fit(model, datamodule=dm)

    # evaluate the model on a test set
    #trainer.test(model)  # uses last-saved model
    # Close wandb run
    wandb.finish()