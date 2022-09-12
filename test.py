import matplotlib.pyplot as plt
import numpy as np

from config import Config
from model import ModelWF
from data_handler import DataHandler
import tensorflow as tf
plt.style.use('mystyle.mlstyle')

config = Config()

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

test_sample         = ["wfs/20t200ma_7t8vov_wfs.txt","wfs/20t200ma_7t8vov_hits.txt","wfs/20t200ma_7t8vov_pars.txt"]
test_data_generator = DataHandler(test_sample,config)

model = ModelWF(config) 
model = model(True)
model.load_weights("models/model_various.h5")

n_batches = 600
n_thresholds = 100
n_classes = config('data','n_classes','int') + 1
pars_keys = config("data","pars","str").split(",")                      #important that this order matches the order in the pars file!


# so many loops, should find a way to optimize this 
tpr, fpr        = np.zeros((n_batches,n_classes,n_thresholds),dtype=np.float32), np.zeros((n_batches,n_classes,n_thresholds),dtype=np.float32)
sample_diff     = np.zeros((n_classes-1),dtype=object)
pars            = np.zeros((n_batches,config("data","batch_size","int"),len(config("data","parameters","str").split(","))),dtype=float)
class_pred      = np.zeros((n_batches,config("data","batch_size","int")),dtype=int)
class_true      = np.zeros((n_batches,config("data","batch_size","int")),dtype=int)
hit_closeby     = []
true_class_instances  = np.zeros((n_classes))
pred_class_instances  = np.zeros((n_classes))

for batch in range(n_batches):
    x, [y_class,y_reg]   = test_data_generator[batch]
    locs                 = test_data_generator.get_locs()  #these are the true hit locations in the waveform
    pars[batch]          = test_data_generator.get_pars()  #these are the parameters of the waveform, vov, gate, etc. 
    pred_class, pred_reg = model.predict(x) 

    pred_class_idx       = np.argmax(pred_class,axis=1) 
    true_class_idx       = np.argmax(y_class,axis=1)  
    class_pred[batch]    = pred_class_idx
    class_true[batch]    = true_class_idx 

    #closeby hit, for now only with n=2
    n2 = true_class_idx == 2
    hit_closeby.append([locs[n2], np.asarray(pred_class_idx[n2] == true_class_idx[n2]).astype(int) ] )

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
                sample_diff[cl-1] = np.append(sample_diff[cl-1], pred_reg.reshape(len(pred_reg))[cl_mask] - y_reg.numpy()[cl_mask])        

            if cl not in true_class_idx: continue
            over_threshold = np.zeros(true_class_idx.shape)                                                                                                           

            # if a prediciton isn't above the threshold, mark as negative (bg), if it is, mark as true (hit)
            over_threshold[pred_class[:,cl]<=threshold] = 0 
            over_threshold[pred_class[:,cl]>threshold]  = 1

            # for multiclass, negative means anythin but cl 
            tp = len(np.where((over_threshold==1) & (true_class_idx==cl) )[0])     # pred_class isn't bg, it wasn't           -> true positive 
            fp = len(np.where((over_threshold==1) & (true_class_idx!=cl) )[0])     # pred_class isn't bg, was                 -> false positive 
            tn = len(np.where((over_threshold==0) & (true_class_idx!=cl) )[0])     # both pred and true are bg                -> true negative
            fn = len(np.where((over_threshold==0) & (true_class_idx==cl) )[0])     # pred is bg but true isn't                -> false negative

            assert tp + fp + tn + fn == int(config('data','batch_size','int')), "total doesn't add up"
            tpr[batch,cl,i] = (tp / (tp+fn)) 
            fpr[batch,cl,i] = (fp / (fp+tn)) 

fpr  = np.average(fpr,axis=0)
tpr  = np.average(tpr,axis=0)
pars = pars.reshape(int(pars.shape[0]*pars.shape[1]),len(config("data","parameters","str").split(",")))
class_true = class_true.reshape(int(class_true.shape[0]*class_true.shape[1]))
class_pred = class_pred.reshape(int(class_pred.shape[0]*class_pred.shape[1]))

# fake hits rate
s=0
for v in [7,8]:
    for g in np.unique(pars[:,-1]):
        mask = (class_true==0) & (pars[:,-1]==g) & (pars[:,0]==v)                                         
        plt.scatter(g,len(class_pred[mask & (class_pred==1)])/len(class_true[mask]),marker='x',color='blue' if v==7 else 'red' ,label = str(v)+" VoV" if s in [0,4] else None)
        s += 1
plt.xlabel('MA Gate [ns]')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#plot single hit efficiency for differnt vov, gates
# group by par, for each par find true==1 / total
# flatten arrays
s=0
for v in [7,8]:
    for g in np.unique(pars[:,-1]):
        mask = (class_true==1) & (pars[:,-1]==g) & (pars[:,0]==v)                   
        plt.scatter(g,len(class_pred[mask & (class_pred==1)])/len(class_true[mask]),marker='x',color='blue' if v==7 else 'red' ,label = str(v)+" VoV" if s in [0,4] else None)
        s += 1
plt.xlabel('MA Gate [ns]')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.ylim(0,1.05)
plt.grid(True)
plt.show()

#plot when two hits were identified correctly as a function of the separation, when is 100% efficiency achieved
hit_closeby = np.asarray(hit_closeby) # contains separations at 0, 1 or 0s at 1
separations = []
accs        = []
for hit2,ones in zip(hit_closeby[:,0],hit_closeby[:,1]): 
    for hit,one in zip(hit2,ones): 
        separations.append(np.diff(hit)) # lots of loops, change this
        accs.append(one)
separations = np.asarray(separations)
accs        = np.asarray(accs)

# digitze the array by separations and calculate the percentage of trues at each bin 
max_separation = 200
nbins          = 30
separation_bins = np.linspace(0,max_separation,nbins)
acc_per_bin_separation = np.zeros(len(separation_bins))
idxs = np.digitize(separations,separation_bins)
for i in range(nbins): 
    idxs = idxs.reshape(idxs.shape[0])
    in_bin = accs[i==idxs] 
    if len(in_bin)!=0:
        acc_per_bin_separation[i] = len(in_bin[in_bin==1]) / len(in_bin)                   ## acc per bin is simply num of 1s divided by total in each bin

plt.plot(10*separation_bins,100*acc_per_bin_separation)
plt.ylabel("Accuracy [\%]")
plt.xlabel("Peak Separation [ns]")
plt.grid(True)
plt.show()

# Roc Curves 
for cl in range(1,n_classes):
    roc_area = float(np.abs(round(np.trapz(tpr[cl],x=fpr[cl]),2)))
    print(str(roc_area))
    plt.step(fpr[cl],tpr[cl],label=r"$N_{{{}}}$".format(cl))
    #plt.step(fpr[cl],tpr[cl],label=r"$N_{{{}}},\int ROC = {{{}}}$".format(cl,str(roc_area)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.grid("True")
plt.xlim(0,1)
plt.ylim(0,1.02)
plt.legend(loc='best')
plt.show()

#plot accuracy, independent of threshold
print(pred_class_instances/true_class_instances,true_class_instances)
plt.bar(np.arange(n_classes),100*pred_class_instances/true_class_instances,width =0.6,alpha=0.8)
plt.hlines(y=100,xmin=-0.8,xmax=n_classes+0.05,linestyles='--',colors='r')
plt.xlabel(r"$N_{hits}$")
plt.ylabel("Accuracy [%]")
plt.xlim(-0.8,n_classes+0.05)
plt.ylim(0,110)
#plt.grid(True)
plt.show()

#plot regression performance
for cl in range(n_classes-1):
    plt.hist(sample_diff[cl].flatten()*config("data","sequence_size","int"),bins=50,range=(-500,500),histtype='step',label=r"$N_{{{}}}$".format(cl+1))
    print(np.std(sample_diff[cl].flatten()*config("data","sequence_size","int")))
    #n,bins,_ = plt.hist(sample_diff.flatten()*config("data","sequence_size","int"),bins=50,range=(-500,500),histtype='step',label="N_{{{}}}".format(cl))
plt.xlabel("True - Reconstructed")
plt.ylabel("Number of Waveforms")
plt.legend()
plt.grid(True)
plt.show()
