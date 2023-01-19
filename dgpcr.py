import tensorflow as tf
import numpy as np
import time

from gpflow.params import DataHolder, Minibatch, Parameter
from gpflow import autoflow, params_as_tensors, ParamList, transforms
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow.likelihoods import Gaussian
from gpflow import settings
float_type = settings.float_type

from doubly_stochastic_dgp.utils import reparameterize

from doubly_stochastic_dgp.utils import BroadcastingLikelihood
from doubly_stochastic_dgp.layer_initializations import init_layers_linear
from doubly_stochastic_dgp.layers import GPR_Layer, SGPMC_Layer, GPMC_Layer, SVGP_Layer


class DGP_Base_CR(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """
    def __init__(self, X, Y, likelihood, layers,
                 minibatch_size=None,
                 num_samples=1, num_data=None,
                 **kwargs):
        Model.__init__(self, **kwargs)
        self.num_samples = num_samples

        self.num_data = num_data or X.shape[0]

        if minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)

        self.likelihood = BroadcastingLikelihood(likelihood)

        self.layers = ParamList(layers)

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmean, Fvar = self._build_predict(X, full_cov=False, S=self.num_samples)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        return tf.reduce_mean(var_exp, 0)  # N, D

    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

class DGPCR(DGP_Base_CR):
    """
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer
    for CROWDSOURCING.

    """
    def __init__(self, X, Y, Z, kernels, likelihood,
                 num_outputs=None, num_latent=None,
                 mean_function=Zero(), minibatch_size=None,  # the final layer mean function,
                 white=True,  alpha=None, alpha_tilde=None, **kwargs):
        class_keys = np.unique(np.concatenate([y[:,1] for y in Y])) # unique identifiers for classes
        num_classes = len(class_keys)
        num_latent = num_latent or num_classes
        num_outputs = num_outputs or num_classes

        layers = init_layers_linear(X, None, Z, kernels,
                                    num_outputs=num_outputs,
                                    mean_function=mean_function,
                                    white=white)
        DGP_Base_CR.__init__(self, X, None, likelihood, layers, minibatch_size=minibatch_size, **kwargs)

        self.num_latent = num_latent
        self.class_keys = class_keys
        self.num_classes = num_classes
        self.annot_keys = np.unique(np.concatenate([y[:,0] for y in Y])) # unique identifiers for annotators
        self.num_annotators = len(self.annot_keys)

        #### Initializing minibatches or placeholders (and the associated idxs_mb to slice q_unn, which seemingly cannot be wrapped in a minibatch or placeholder, maybe because it is already a gpflow Parameter).
        startTime = time.time()
        Y_idxs = np.array([np.stack((np.array([np.flatnonzero(v==self.annot_keys)[0] for v in y[:,0]]),
                                     np.array([np.flatnonzero(v==self.class_keys)[0] for v in y[:,1]])), axis=1) for y in Y]) # same as Y but contains indexes over annot_keys and class_keys
        S = np.max([v.shape[0] for v in Y_idxs]) # Maximum number of annotations for a single instance
        Y_idxs_cr = np.array([np.concatenate((y,-1*np.ones((S-y.shape[0],2))),axis=0) for y in Y_idxs]).astype(np.int16) # Padding Y_idxs with -1 to create a NxSx2 array.
        if minibatch_size is None:
            self.Y_idxs_cr = DataHolder(Y_idxs_cr)
            self.idxs_mb = DataHolder(np.arange(self.num_data))
        else:
            self.Y_idxs_cr = Minibatch(Y_idxs_cr, batch_size=minibatch_size, seed=0)
            self.idxs_mb = Minibatch(np.arange(self.num_data), batch_size=minibatch_size, seed=0)
        print("Time taken in Y_idxs creation:", time.time()-startTime)

        #### Initializing the approximate posterior q(z). q_unn is NxK, and constrained to be positive. "_unn" denotes unnormalized: rows must be normalized to obtain the posterior q(z). We could have also implemented a gpflow transform that constrains sum to 1 by rows. Notice that q_unn initialization is based on the annotations.
        startTime = time.time()
        q_unn = np.array([np.bincount(y[:,1], minlength=self.num_classes) for y in Y_idxs])
        q_unn = q_unn + np.ones(q_unn.shape)
        q_unn = q_unn/np.sum(q_unn,axis=1,keepdims=True)
        self.q_unn = Parameter(q_unn,transform=transforms.positive) # N x K
        print("Time taken in q_unn initialization:", time.time()-startTime)

        #### Initializing alpha (fixed) and alpha_tilde (trainable). Both have size AxKxK
        if alpha is None:
            alpha = np.ones((self.num_annotators,self.num_classes,self.num_classes), dtype=settings.float_type)
        self.alpha = Parameter(alpha, transform=transforms.positive, trainable=False)
        startTime = time.time()
        alpha_tilde = self._init_behaviors(q_unn, Y_idxs)
        print("Time taken in alpha_tilde initialization:", time.time()-startTime)
        self.alpha_tilde = Parameter(alpha_tilde,transform=transforms.positive)

    def _init_behaviors(self, probs, Y_idxs):
        alpha_tilde = np.ones((self.num_annotators,self.num_classes,self.num_classes))/self.num_classes
        counts = np.ones((self.num_annotators,self.num_classes))
        for n in range(len(Y_idxs)):
            for a,c in zip(Y_idxs[n][:,0], Y_idxs[n][:,1]):
                alpha_tilde[a,c,:] += probs[n,:]
                counts[a,c] += 1
        alpha_tilde=alpha_tilde/counts[:,:,None]
        alpha_tilde = (counts/np.sum(counts,axis=1,keepdims=True))[:,:,None]*alpha_tilde
        return alpha_tilde/np.sum(alpha_tilde,axis=1,keepdims=True)

    @params_as_tensors
    def build_annot_KL(self): # Computes KL div between q(R) and p(R) [5th term in the paper]
        alpha_diff = self.alpha_tilde-self.alpha
        KL_annot=(tf.reduce_sum(tf.multiply(alpha_diff,tf.digamma(self.alpha_tilde)))-
        tf.reduce_sum(tf.digamma(tf.reduce_sum(self.alpha_tilde,1))*tf.reduce_sum(alpha_diff,1))+
        tf.reduce_sum(tf.lbeta(tf.matrix_transpose(self.alpha))
        -tf.lbeta(tf.matrix_transpose(self.alpha_tilde))))
        return KL_annot

    @params_as_tensors
    def _build_likelihood(self): # It returns the ELBO in the paper

        KL = tf.reduce_sum([layer.KL() for layer in self.layers]) # [4th term in the paper]
        KL_annot = self.build_annot_KL() # [5th term in the paper]

        #### ENTROPY OF Q COMPONENT [3rd term in the paper]
        q_unn_mb = tf.gather(self.q_unn,self.idxs_mb)   # N x K
        q_mb = tf.divide(q_unn_mb, tf.reduce_sum(q_unn_mb, axis=1, keepdims=True)) # N x K
        qentComp = tf.reduce_sum(tf.multiply(q_mb,tf.log(q_mb)))

        #### LIKELIHOOD COMPONENT [2nd term in the paper]
        fmean, fvar = self._build_predict(self.X, full_cov=False, S=self.num_samples)
        tensors_list = [tf.reduce_mean(self.likelihood.variational_expectations(fmean, fvar, c*tf.ones((tf.shape(self.X)[0],1),dtype=tf.int32)),0) for c in np.arange(self.num_classes)]
        tnsr_lik = tf.concat(tensors_list,-1)  # NxK
        lhoodComp = tf.reduce_sum(tf.multiply(q_mb,tnsr_lik))

        #### CROWDSOURCING COMPONENT [1st term in the paper]
        expect_log = tf.digamma(self.alpha_tilde)-tf.digamma(tf.reduce_sum(self.alpha_tilde,1,keepdims=True)) # A x K x K
        tnsr_expCrow = tf.gather_nd(expect_log, tf.cast(self.Y_idxs_cr, tf.int32)) # N x S x K (on GPU, indexes -1 return 0, see documentation for tf.gather_nd. On CPU, an error would be raised.)
        crComp = tf.reduce_sum(tf.multiply(tnsr_expCrow, tf.expand_dims(q_mb,1)))

        scale = tf.cast(self.num_data, settings.float_type)/tf.cast(tf.shape(self.X)[0], settings.float_type)
        self.decomp = [lhoodComp,crComp,qentComp,KL,KL_annot,scale]
        return ((lhoodComp+crComp-qentComp)*scale-KL-KL_annot)
