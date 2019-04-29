from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
from autograd import grad
from autograd.extend import notrace_primitive

@notrace_primitive
def resampling(w, rs):
    """
    Stratified resampling with "nograd_primitive" to ensure autograd 
    takes no derivatives through it.
    """
    N = w.shape[0]
    bins = np.cumsum(w)
    ind = np.arange(N)
    u = (ind  + rs.rand(N))/N
    
    return np.digitize(u, bins)

def vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, verbose=False, adapt_resamp=False):
    """
    Estimate the VSMC lower bound. Amenable to (biased) reparameterization 
    gradients.

    .. math::
        ELBO(\theta,\lambda) =
        \mathbb{E}_{\phi}\left[\nabla_\lambda \log \hat p(y_{1:T}) \right]

    Requires an SMC object with 2 member functions:
    -- sim_prop(t, x_{t-1}, y, prop_params, model_params, rs)
    -- log_weights(t, x_t, x_{t-1}, y, prop_params, model_params)
    """
    # Extract constants
    T = y.shape[0]
    Dx = smc_obj.Dx
    N = smc_obj.N

    # Initialize SMC
    X = np.zeros((N,Dx))
    Xp = np.zeros((N,Dx))
    logW = np.zeros(N)
    W = np.exp(logW)
    W /= np.sum(W)
    logZ = 0.
    ESS = 1./np.sum(W**2)/N

    for t in range(T):
        # Resampling
        if adapt_resamp:
            if ESS < 0.5:
                ancestors = resampling(W, rs)
                Xp = X[ancestors]
                logZ = logZ + max_logW + np.log(np.sum(W)) - np.log(N)
                logW = np.zeros(N)
            else:
                Xp = X
        else:
            if t > 0:
                ancestors = resampling(W, rs)
                Xp = X[ancestors]
            else:
                Xp = X

        # Propagation
        X = smc_obj.sim_prop(t, Xp, y, prop_params, model_params, rs)

        # Weighting
        if adapt_resamp:
            logW = logW + smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)
        else:
            logW = smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)
        max_logW = np.max(logW)
        W = np.exp(logW-max_logW)
        if adapt_resamp:
            if t == T-1:
                logZ = logZ + max_logW + np.log(np.sum(W)) - np.log(N)
        else:
            logZ = logZ + max_logW + np.log(np.sum(W)) - np.log(N)
        W /= np.sum(W)
        ESS = 1./np.sum(W**2)/N
    if verbose:
        print('ESS: '+str(ESS))
    return logZ

def sim_q(prop_params, model_params, y, smc_obj, rs, verbose=False):
    """
    Simulates a single sample from the VSMC approximation.

    Requires an SMC object with 2 member functions:
    -- sim_prop(t, x_{t-1}, y, prop_params, model_params, rs)
    -- log_weights(t, x_t, x_{t-1}, y, prop_params, model_params)
    """
    # Extract constants
    T = y.shape[0]
    Dx = smc_obj.Dx
    N = smc_obj.N

    # Initialize SMC
    X = np.zeros((N,T,Dx))
    logW = np.zeros(N)
    W = np.zeros((N,T))
    ESS = np.zeros(T)

    for t in range(T):
        # Resampling
        if t > 0:
            ancestors = resampling(W[:,t-1], rs)
            X[:,:t,:] = X[ancestors,:t,:]

        # Propagation
        X[:,t,:] = smc_obj.sim_prop(t, X[:,t-1,:], y, prop_params, model_params, rs)

        # Weighting
        logW = smc_obj.log_weights(t, X[:,t,:], X[:,t-1,:], y, prop_params, model_params)
        max_logW = np.max(logW)
        W[:,t] = np.exp(logW-max_logW)
        W[:,t] /= np.sum(W[:,t])
        ESS[t] = 1./np.sum(W[:,t]**2)

    # Sample from the empirical approximation
    bins = np.cumsum(W[:,-1])
    u = rs.rand()
    B = np.digitize(u,bins)

    if verbose:
        print('Mean ESS', np.mean(ESS)/N)
        print('Min ESS', np.min(ESS))
        
    return X[B,:,:]
