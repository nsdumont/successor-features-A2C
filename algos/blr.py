import numpy as np

# BLR code written under the influence of 
# https://maxhalford.github.io/blog/bayesian-linear-regression/

class BayesianLinearRegression:
    def __init__(self, size_in, size_out=1, alpha=0.01, beta=None):
        self.input_dim = size_in
        self.output_dim = size_out
        self.S_inv = alpha * np.eye(size_in)
        self.m = np.zeros((size_in,1))
        self.beta = beta
        self.S = np.linalg.pinv(self.S_inv)
        self.rank_one_updates = True
    ### end __init__

    def update(self, phis:np.ndarray, ts:np.ndarray):
        '''
        Compute one-step update of bayesian linear regression
        '''
        assert phis.shape[1] == self.input_dim, f'Expected input shape ({ts.shape[0]}, {self.input_dim}), got {phis.shape}'
        assert len(ts.shape) > 1 and ts.shape[1] == self.output_dim, f'Expected output shape ({phis.shape[0]}, 1), got {ts.shape}'
#         assert np.isfinite(phis).all(), 'Received a non-finite value'

        if self.beta is None:
#             assert len(ts) > 1, f'Expected more than one initial example, got {len(ts)}.'
            if len(ts) > 1:
                var_ts = np.var(ts)
            else:
                var_ts = np.abs(np.copy(ts[0]))
            self.beta = 1. / var_ts
        S_inv = self.S_inv + self.beta * np.dot(phis.T, phis)
        
        S = np.copy(self.S)
        if self.rank_one_updates:
            for i in range(phis.shape[0]):
                phi = np.atleast_2d(phis[i,:])  * np.sqrt(self.beta)
                scale = (1 + phi @ S @ phi.T)
                S = S - (S @ phi.T @ phi @ S) / scale
        else:
            S = np.linalg.pinv(S_inv)
        x = self.beta * np.dot(phis.T, ts)
        assert x.shape == (self.input_dim, 1), f'Mean update should be shape {self.input_dim, 1} was {x.shape}'
        
        self.m = S @ (self.S_inv @ self.m + x)
        self.S_inv = S_inv
        self.S = S

        assert self.m.shape[0] == self.input_dim and self.m.shape[1] == 1

    def predict(self, phi):
        var = (1. / self.beta) + np.einsum('ij,ij->i', phi, np.dot(phi, self.S.T))
        return np.dot(self.m.T, phi.T), var

    def sample(self):
        phi_init = None
        try:
            phi_init = np.atleast_2d(np.random.multivariate_normal(self.m.flatten(), 
                                                                   self.S).reshape(-1,1)
                                    )
        except np.linalg.LinAlgError as e:
            print(e)
            phi_init = (-self.S_inv @ self.m).reshape(-1,1)
        assert phi_init.ndim == 2
        assert phi_init.shape[1]
        return phi_init
    ### end sample
### end class BayesianLinearRegression
