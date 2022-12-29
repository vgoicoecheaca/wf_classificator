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


n_thresholds  = config('testing','n_thresholds','int') 
n_classes     = 2
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
y_pred_class             = y_pred_class.reshape(len(y_pred_class))
y_true_class             = y_true_class.reshape(len(y_true_class))
true_locs                = test_data_generator.get_locs()
pars                     = test_data_generator.get_pars()

y_pred_class_idx = y_pred_class
y_pred_class_idx[y_pred_class>=0.5] = 1
y_pred_class_idx[y_pred_class<0.5] = 0

#closeby hit, for now only with n=2
#hit_closeby.append([true_locs[y_pred_class==1], np.asarray(y_pred_class[y_pred_class==1] == y_true_class[y_pred_classs==1]).astype(int)])

#accuracy independent of the threshold
v, c = np.unique(y_true_class,return_counts=True)  
true_class_instances[v.astype(int)] += c
v, c = np.unique(y_pred_class_idx[y_pred_class_idx == y_true_class],return_counts=True)
pred_class_instances[v.astype(int)] += c

#ROC is true positive rate vs false positive rate
# recall is true positive / true positive + false negatives
#thresholds = np.linspace(0,1,n_thresholds)
#for i,threshold in enumerate(thresholds):
#    for cl in np.unique(y_pred_class):
#        # regression performance is simply the difference between the predicted and truth beggining of pulse                              
#        #if i==0: # otherwise it'll be repeated for each threshold.... 
#        #    cl_mask = y_true_class == cl 
#        #    sample_diff[cl-1] = np.append(sample_diff[cl-1],y_pred_reg[cl_mask] - y_true_reg[cl_mask])        
#
#        #if cl not in true_class_idx: continue
#        over_threshold = np.zeros(y_pred_class.shape)                                                                                                           
#
#        # if a prediciton isn't above the threshold, mark as negative (bg), if it is, mark as true (hit)
#        over_threshold[y_pred_class<=threshold] = 0 
#        over_threshold[y_pred_class>threshold]  = 1
# 
#        # for multiclass, negative means anythin but cl 
#        tp = len(np.where((over_threshold==1) & (y_true_class==cl) )[0])     # pred_class isn't cl, it wasn't           -> true positive 
#        fp = len(np.where((over_threshold==1) & (y_true_class!=cl) )[0])     # pred_class isn't cl, was                 -> false positive 
#        tn = len(np.where((over_threshold==0) & (y_true_class!=cl) )[0])     # both pred and true are cl                -> true negative
#        fn = len(np.where((over_threshold==0) & (y_true_class==cl) )[0])     # pred is cl but true isn't                -> false negative
#
#        assert tp + fp + tn + fn == len(y_pred_class)
#
#        tpr[cl,i] = (tp / (tp+fn)) 
#        fpr[cl,i] = (fp / (fp+tn)) 

#Plotting 
print(pred_class_instances/true_class_instances)
print(y_pred_class)
print(y_true_class)
plotter = Plotter(config)
#plotter.roc(fpr,tpr)
plotter.acs(pred_class_instances,true_class_instances)
exit()
#plotter.regs(sample_diff)
#plotter.fake(true_class_idx,pred_class_idx,pars)
#plotter.single_hit_eff(true_class_idx,pred_class_idx,pars)
#plotter.double_separation(hit_closeby)

