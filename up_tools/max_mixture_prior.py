"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All
rights reserved.

This software is provided for research purposes only.

By using this software you agree to the terms of the SMPLify license here:

     http://smplify.is.tue.mpg.de/license
"""
# pylint: disable=no-member, invalid-name, attribute-defined-outside-init
import os
import numpy as np
import chumpy as ch

class MaxMixtureComplete(ch.Ch):

    """Define the MaxMixture class."""

    dterms = 'x', 'means', 'precs', 'weights'

    def on_changed(self, which):
        # setup means, precs and loglikelihood expressions
        if 'means' in which or 'precs' in which or 'weights' in which:
            # This is just the mahalanobis part.
            self.loglikelihoods = [np.sqrt(0.5) * (self.x - m).dot(s)
                                   for m, s in zip(self.means, self.precs)]


        if 'x' in which:
            # start = time.time()
            self.min_component_idx = np.argmin([(logl**2).sum().r[0] -np.log(w[0])
                                                for logl, w in zip(self.loglikelihoods,
                                                                   self.weights)])

    def compute_r(self):
        # pylint: disable=unsubscriptable-object
        min_w = self.weights[self.min_component_idx]
        # Add the sqrt(-log(weights)).
        return ch.concatenate((self.loglikelihoods[self.min_component_idx].r,
                               np.sqrt(-np.log(min_w))))


    def compute_dr_wrt(self, wrt):
        # Returns 69 x 72, when wrt is 69D => return 70x72 with empty last for
        # when returning 70D
        # Extract the data, rows cols and data, new one with exact same values
        # but with size one more rows)
        import scipy.sparse as sp

        dr = self.loglikelihoods[self.min_component_idx].dr_wrt(wrt)
        if dr is not None:
            Is, Js, Vs = sp.find(dr)
            dr = sp.csc_matrix((Vs, (Is, Js)), shape=(dr.shape[0]+1, dr.shape[1]))

        return dr


# pylint: disable=too-few-public-methods
class MaxMixtureCompleteWrapper(object):

    """Convenience wrapper to match interface spec."""

    def __init__(self, means, precs, weights, prefix):
        self.means = means
        self.precs = precs # Already "sqrt"ed
        self.weights = weights
        self.prefix = prefix

    def __call__(self, x):
        return(MaxMixtureComplete(x=x[self.prefix:], means=self.means,
                                  precs=self.precs, weights=self.weights))



class MaxMixtureCompletePrior(object):

    """Prior density estimation."""

    def __init__(self, n_gaussians=6, prefix=3):
        self.n_gaussians = n_gaussians
        self.prefix = prefix
        self.prior = self.create_prior_from_cmu()

    def create_prior_from_cmu(self):
        """Get the prior from the CMU motion database."""
        from os.path import realpath
        import cPickle as pickle
        with open(os.path.join(os.path.dirname(__file__),
                               '..', 'models', '3D',
                               'gmm_best_%02d.pkl' % self.n_gaussians)) as f:
            gmm = pickle.load(f)

        precs = ch.asarray([np.linalg.inv(cov) for cov in gmm.covars_])
        chols = ch.asarray([np.linalg.cholesky(prec) for prec in precs])

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in gmm.covars_])
        const = (2*np.pi)**(69/2.)

        self.weights = ch.asarray(gmm.weights_ / (const * (sqrdets/sqrdets.min())))

        return(MaxMixtureCompleteWrapper(means=gmm.means_, precs=chols,
                                         weights=self.weights, prefix=self.prefix))

    def get_gmm_prior(self):
        """Getter implementation."""
        return self.prior
