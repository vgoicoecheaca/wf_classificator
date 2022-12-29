
import matplotlib.pyplot as plt
import numpy as np 

plt.style.use('mystyle.mlstyle')

class Plotter():
    def __init__(self,config):
        self.n_classes     = 2
        self.sequence_size = config("data","sequence_size","int")
        self.pars          = config("data","parameters","str").split(",")

    def roc(self,fpr,tpr):
        for cl in range(1,self.n_classes):
            roc_area = float(np.abs(round(np.trapz(tpr[cl],x=fpr[cl]),2)))
            print("ROC Area of ",str(cl),str(roc_area))
            plt.step(fpr[cl],tpr[cl],label=r"$N_{{{}}}$".format(cl))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.grid("True")
        plt.xlim(0,1)
        plt.ylim(0,1.02)
        plt.legend(loc='best')
        plt.savefig("plots/roc.pdf")
        plt.clf()

    def acs(self,pred_class_instances,true_class_instances):
        #plot accuracy, independent of threshold
        print("Accs",pred_class_instances/true_class_instances,true_class_instances)
        plt.bar(np.arange(self.n_classes),100*pred_class_instances/true_class_instances,width =0.6,alpha=0.8)
        plt.hlines(y=100,xmin=-0.8,xmax=self.n_classes+0.05,linestyles='--',colors='r')
        plt.xlabel(r"$N_{hits}$")
        plt.ylabel("Accuracy [\%]")
        plt.xlim(-0.8,self.n_classes+0.05)
        plt.ylim(0,110)
        plt.savefig("plots/accs.pdf")
        plt.clf()

    def regs(self,sample_diff):
        #plot regression performance
        for cl in range(self.n_classes-1):
            plt.hist(sample_diff[cl].flatten()*self.sequence_size,bins=50,range=(-500,500),histtype='step',label=r"$N_{{{}}}$".format(cl+1))
            print(np.std(sample_diff[cl].flatten()*self.sequence_size))
        plt.xlabel("True - Reconstructed")
        plt.ylabel("Number of Waveforms")
        plt.legend()
        plt.grid(True)
        plt.savefig("plots/regs.pdf")
        plt.clf()

    def fake(self,class_true,class_pred,pars):
        # note that this only wors for vov, ma, in this order
        s=0
        vov_idx = int(self.pars.index("vov"))
        g_idx   = int(self.pars.index("ma"))
        for vi,v in enumerate(np.unique(pars[:,vov_idx])):
            for g in np.unique(pars[:,g_idx]):
                mask = (class_true!=0) & (pars[:,g_idx]==g) & (pars[:,vov_idx]==v)                                         # implement automatic way of doing this
                plt.scatter(g,len(class_pred[mask & (class_pred==0)])/len(class_true[mask]),marker='x',color='blue' if vi==0 else 'red' ,label = str(v)+" VoV" if s in [0,10] else None)
                s += 1
        plt.xlabel('MA Gate [ns]')
        plt.ylabel('Fake Hits Freq')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig("plots/fake.pdf")
        plt.clf()

    def single_hit_eff(self,class_true,class_pred,pars):
        # note that this only wors for vov, ma, in this order
        #plot single hit efficiency for differnt vov, gates
        # group by par, for each par find true==1 / total
        s=0
        vov_idx = int(self.pars.index("vov"))
        g_idx   = int(self.pars.index("ma"))
        for vi,v in enumerate(np.unique(pars[:,vov_idx])):
            for g in np.unique(pars[:,g_idx]): 
                mask = (class_true==1) & (pars[:,g_idx]==g) & (pars[:,vov_idx]==v)                                             # implement automatic way of doing this
                plt.scatter(g,len(class_pred[mask & (class_pred==1)])/len(class_true[mask]),marker='x',color='blue' if vi==0 else 'red' ,label = str(v)+" VoV" if s in [0,10] else None)
                s += 1
        plt.xlabel('MA Gate [ns]')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.ylim(0,1.05)
        plt.grid(True)
        plt.savefig("plots/single.pdf")
        plt.clf()

    def double_separation(self,hit_closeby):
        #plot when two hits were identified correctly as a function of the separation, when is 100% efficiency achieved
        hit_closeby = np.asarray(hit_closeby) # contains separations at 0, 1 or 0s at 1
        separations,accs = [],[]
        for hit2,ones in zip(hit_closeby[:,0],hit_closeby[:,1]): 
            for hit,one in zip(hit2,ones): 
                separations.append(np.diff(hit)) # lots of loops, change this
                accs.append(one)
        separations, accs = np.asarray(separations), np.asarray(accs)
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
        plt.plot(10*separation_bins,100*acc_per_bin_separation,"x--")
        plt.ylabel("Accuracy [\%]")
        plt.xlabel("Peak Separation [ns]")
        plt.grid(True)
        plt.savefig("plots/separation.pdf")
        plt.clf()

    def waveform_vis(self,x,y_class_true,y_reg_true,pred_class,pred_reg):
        for i in range(len(x)):
            plt.plot(x[i],color='blue')
            print("Pred Hits",np.argmax(pred_class[i]))
            print("True Hits",np.argmax(y_class_true[i]))
            print("Pred pos",pred_reg[i])
            print("True pos",y_reg_true[i])
            plt.vlines(x=pred_reg[i]*self.sequence_size,ymin=0,ymax=1,color='orange',alpha=0.8,label="start reco")
            plt.vlines(x=y_reg_true[i]*self.sequence_size,ymin=0,ymax=1,color='green',alpha=0.8,label="start truth")
            plt.text(x=self.sequence_size*0.8,y=0.85,s="True hits = "+str(np.argmax(y_class_true[i])),color='green')
            plt.text(x=self.sequence_size*0.8,y=0.80,s="Reco hits = "+str(np.argmax(pred_class[i])),color='orange')
            plt.legend(loc='best')
            plt.savefig("plots/wf_reco"+str(i)+".pdf")
            plt.clf()
