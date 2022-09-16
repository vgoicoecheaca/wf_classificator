import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import Config
from model import ModelWF
from data_handler import DataHandler
from plotter import Plotter


plt.style.use('mystyle.mlstyle')
config      = Config()
DISABLE_GPU = config('training','disable_gpu','bool')

if DISABLE_GPU:
    try:
        # Disable all GPUS
        visible_devices = tf.config.get_visible_devices()
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


n_batches     = config('testing','batches','int') 
n_thresholds  = config('testing','n_thresholds','int') 
n_classes     = config('data','n_classes','int') + 1
pars_keys     = config("data","parameters","str").split(",")                      #important that this order matches the order in the pars file!
batch_size    = config('data','batch_size','int') 

data_path           = "wfs/"
test_sample         = [data_path+i for i in config("testing","data","str").split(",")]
test_data_generator = DataHandler(test_sample,config)

x_test, [y_true_class,y_true_reg] = test_data_generator()

model = ModelWF(config) 
model = model(True)
model.load_weights("models/"+config("testing","model","str"))

tpr, fpr              = np.zeros((n_classes,n_thresholds),dtype=np.float32), np.zeros((n_classes,n_thresholds),dtype=np.float32)
sample_diff           = np.zeros((n_classes-1),dtype=object)
hit_closeby           = []
true_class_instances  = np.zeros((n_classes))
pred_class_instances  = np.zeros((n_classes))

y_pred_class, y_pred_reg = model.predict(x_test)
true_locs                = test_data_generator.get_locs()
pars                     = test_data_generator.get_pars()
pred_class_idx           = np.argmax(y_pred_class,axis=1) 
true_class_idx           = np.argmax(y_true_class,axis=1)  

#closeby hit, for now only with n=2
n2 = true_class_idx == 2
hit_closeby.append([true_locs[n2], np.asarray(pred_class_idx[n2] == true_class_idx[n2]).astype(int)])

#accuracy independent of the threshold
v, c = np.unique(true_class_idx,return_counts=True)  
true_class_instances[v] +=  c 
v, c = np.unique(pred_class_idx[pred_class_idx == true_class_idx],return_counts=True)
pred_class_instances[v] += c 

#ROC is true positive rate vs false positive rate
# recall is true positive / true positive + false negatives
thresholds = np.linspace(0,1,n_thresholds)
for i,threshold in enumerate(thresholds):
    for cl in range(1,n_classes):
        # regression performance is simply the difference between the predicted and truth beggining of pulse                              
        if i==0: # otherwise it'll be repeated for each threshold.... 
            cl_mask = true_class_idx  == cl 
            sample_diff[cl-1] = np.append(sample_diff[cl-1],y_pred_reg[cl_mask] - y_true_reg[cl_mask])        

        if cl not in true_class_idx: continue
        over_threshold = np.zeros(true_class_idx.shape)                                                                                                           

        # if a prediciton isn't above the threshold, mark as negative (bg), if it is, mark as true (hit)
        over_threshold[y_pred_class[:,cl]<=threshold] = 0 
        over_threshold[y_pred_class[:,cl]>threshold]  = 1
 
        # for multiclass, negative means anythin but cl 
        tp = len(np.where((over_threshold==1) & (true_class_idx==cl) )[0])     # pred_class isn't cl, it wasn't           -> true positive 
        fp = len(np.where((over_threshold==1) & (true_class_idx!=cl) )[0])     # pred_class isn't cl, was                 -> false positive 
        tn = len(np.where((over_threshold==0) & (true_class_idx!=cl) )[0])     # both pred and true are cl                -> true negative
        fn = len(np.where((over_threshold==0) & (true_class_idx==cl) )[0])     # pred is cl but true isn't                -> false negative

        assert tp + fp + tn + fn == len(y_pred_class)

        tpr[cl,i] = (tp / (tp+fn)) 
        fpr[cl,i] = (fp / (fp+tn)) 

#Plotting 
plotter = Plotter(config)
#plotter.roc(fpr,tpr)
#plotter.acs(pred_class_instances,true_class_instances)
#plotter.regs(sample_diff)
plotter.fake(true_class_idx,pred_class_idx,pars)
plotter.single_hit_eff(true_class_idx,pred_class_idx,pars)
plotter.double_separation(hit_closeby)

