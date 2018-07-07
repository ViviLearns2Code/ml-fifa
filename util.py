'''
    File name: util.py
    Author: vivilearnstocode
    Last modified: 07/07/2018
    Python version: 3.6
'''
import scipy as sc
import numpy as np

class MyUtil(object):
    def calc_nb_param(self,mu,alpha):
        r=1/alpha
        p=r/(mu+r)
        return p,r

    def calc_nb_probs(self,muA,alphaA,muB,alphaB):
        pA,rA=self.calc_nb_param(muA,alphaA)
        pB,rB=self.calc_nb_param(muB,alphaB)
        nbA = np.vectorize(sc.stats.nbinom)(rA[:,0],pA[:,0])
        nbB = np.vectorize(sc.stats.nbinom)(rB[:,0],pB[:,0])
        pdfA = np.array(list(map(lambda x:x.pmf(np.arange(11)),nbA)))
        pdfB= np.array(list(map(lambda x:x.pmf(np.arange(11)),nbB)))
        i=0
        probs=np.zeros((pdfA.shape[0],pdfA.shape[1],pdfA.shape[1]))
        for row in pdfA:
            probs[i,:,:]=np.outer(row,pdfB[i,:])
            i+=1
        return probs

    def tend_acc_nb(self,muA,alphaA,muB,alphaB,y):
        probs=self.calc_nb_probs(muA,alphaA,muB,alphaB)
        return self.multi_tendency(y,probs) 

    def multi_tendency(self,y,y_prob):
        i=0
        result_pred = np.zeros(y.shape[0])
        for row in y_prob:
            winA = np.tril(row).sum()
            winB = np.triu(row).sum()
            draw = np.trace(row)
            if winA >= winB and winA >= draw:
                result_pred[i] = 1
            if winB >= winA and winB >= draw:
                result_pred[i] = -1
            if draw >= winA and draw >= winB:
                result_pred[i] = 0                
            i+=1
        act = np.vectorize(self.encode_tendency)(y[:,0],y[:,1])
        return np.sum(act==result_pred)/y.shape[0]

    def multi_result(self,y_prob,top_n,verbose=False,y=None):
        ''' 
        Checks how often the top n results in y_prob contain the true result
        '''
        N=y_prob.shape[0]
        R=y_prob.shape[1]
        C=y_prob.shape[2]
        count=0
        i=0
        for row in y_prob:
            probs=row.ravel() #concatenate all rows to one long row vector
            idx_ravel = np.argsort(probs)[::-1][:top_n]
            idx_unravel = self.get_index(R,C,idx_ravel)
            probs = probs[idx_ravel].reshape((idx_ravel.shape[0],1))
            if y is None: 
                print("candidates \n",np.concatenate((idx_unravel,probs),axis=1))
                continue
            if np.equal(idx_unravel,y[i,:]).all(axis=1).any():
                # top predictions contain correct one
                if verbose==True:
                    print("right prediction for",y[i,:])
                    print("candidates \n",np.concatenate((idx_unravel,probs),axis=1))
                count+=1
            elif verbose==True:
                print("wrong prediction for",y[i,:])
                print("candidates \n",np.concatenate((idx_unravel,probs),axis=1))
            i+=1
        return count/N

    def single_tendency(self,y,y_prob):
        '''
        Checks how often the top-1 prediction gets the tendency right (win/draw/loss)
        '''
        act = np.vectorize(self.encode_tendency)(y[:,0],y[:,1])
        pred = self.get_top_n(probs=y_prob,top_n=1)
        pred = np.vectorize(self.encode_tendency)(pred[:,0,0],pred[:,0,1])
        return np.sum(act==pred)/y.shape[0]

    def get_top_n(self,probs,top_n=5):
        ''' 
        Get top n most likely results
        
        Input:
        - probs: array of shape(n_samples,n_counts,n_counts) game outcome probabilities for each sample
        - top_n: get top_n most likely results
        
        Returns:
        - result: array of shape(n_samples,top_n,2) top_n most likely game results
        '''
        N=probs.shape[0]
        R=probs.shape[1]
        C=probs.shape[2]
        result = np.zeros((N,top_n,2))
        i=0
        for row in probs:
            prob=row.ravel()
            idx_ravel = np.argsort(prob)[::-1][:top_n]
            idx_unravel = self.get_index(R,C,idx_ravel)
            result[i,:,:] = idx_unravel
            i+=1
        return result

    def get_index(self,rows,cols,idx):
        col_idx = idx % rows
        row_idx = idx // rows  
        idx = np.column_stack((row_idx,col_idx))
        return idx

    def encode_tendency(self,x,y,win=1,draw=0,loss=-1):
        if x > y:
            return win
        elif x == y:
            return draw
        else:
            return loss

    def tend_acc_pois(self,mu,y):
        poA = np.vectorize(sc.stats.poisson)(mu[:,0])
        poB = np.vectorize(sc.stats.poisson)(mu[:,1])
        pdfA_po = np.array(list(map(lambda x:x.pmf(np.arange(11)),poA)))
        pdfB_po= np.array(list(map(lambda x:x.pmf(np.arange(11)),poB)))
        i=0
        probs_po=np.zeros((pdfA_po.shape[0],pdfA_po.shape[1],pdfA_po.shape[1]))
        for row in pdfA_po:
            probs_po[i,:,:]=np.outer(row,pdfB_po[i,:])
            i+=1
        return self.multi_tendency(y,probs_po)

    def calc_pois_probs(self,muA,muB):
        poA = np.vectorize(sc.stats.poisson)(muA)
        poB = np.vectorize(sc.stats.poisson)(muB)
        pdfA = np.array(list(map(lambda x:x.pmf(np.arange(11)),poA)))
        pdfB= np.array(list(map(lambda x:x.pmf(np.arange(11)),poB)))
        i=0
        probs=np.zeros((pdfA.shape[0],pdfA.shape[1],pdfA.shape[1]))
        for row in pdfA:
            probs[i,:,:]=np.outer(row,pdfB[i,:])
            i+=1
        return probs