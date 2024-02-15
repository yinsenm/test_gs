import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import sqrt
import cvxpy as cp


def set_eps_wgt_to_zeros(in_array, eps=1e-4):
    # set small weights to 0 and return a list
    out_array = np.array(in_array)
    out_array[np.abs(in_array) < eps] = 0
    out_array = np.array(out_array) / np.sum(out_array)
    return out_array

def get_strats(cov_name, cov_func, cov_config, list_of_vols: list):
    dict_others = {
        "%s-%s" % (strategy_name, cov_name): {"strategy": strategy_name,
                                              "config": {"cov": cov_func, "cov_params": cov_config}}
        for strategy_name in ["mdp", "rp", "mvp", "mr"]
    }

    dict_mvos = {
        "mvo=%02dpct-%s" % (vol, cov_name): {"strategy": "mvo", "config": {"vol_target": vol / 100. / sqrt(12), "cov": cov_func, "cov_params": cov_config}}
        for vol in list_of_vols
    }
    return {**dict_others, **dict_mvos}

class PortfolioStrategy:
    def __init__(self,
                 strategy: str,
                 config: dict = None,
                 min_weight: float = 0,
                 max_weight: float = 1,
                 txn_cost: float = None,
                 solver: str = "MOSEK"):
        """
        :param strategy: name of the strategy
        :param config: config of the strategy in dictionary
        :param min_weight: constraint on minimum weight on asset
        :param max_weight: constraint on maximum weight on asset
        :param txn_cost: cost of transaction fee and slippage in bps or 0.01%
        """
        assert strategy in ["ew", "mvo", "mvp", "mr", "mdp", "ivw", "rp", "benchmark"], \
            "strategy must be one of ew, mvo, mvp, mr, mdp, ivw, rp and benchmark strategies."
        self.strategy = strategy
        self.config = config
        self.min_weight = min_weight
        self.max_weight = max_weight
        if txn_cost is not None:
            self.txn_cost = txn_cost / 10000
        else:
            self.txn_cost = 0

        assert 0 <= self.min_weight < 1, "min_weight in [0, 1)"
        assert 0 < self.max_weight <= 1, "max_weight in (0, 1]"
        assert self.min_weight < self.max_weight, "min_weight < max_weight"
        assert self.txn_cost >= 0, "transaction penalty must be larger or equal to 0"

        self.prev_weights = None  # previous weights of portfolio
        self.init_weights = None  # initial weights of portfolio optimizer
        self.mu_vec = None  # expected return
        self.cov_mat = None  # save cov_mat
        self.solver = solver

    def get_weights(self, df_rets: pd.DataFrame) -> dict:
        if self.strategy == "ew":
            return self._ew_func(df_rets)
        elif self.strategy == "ivw":
            return self._ivw_func(df_rets)
        elif self.strategy == "mvo":  # mean variance optimization
            return self._mvo_func(df_rets)
        elif self.strategy == "mr":   # maximize return
            return self._mr_func(df_rets)
        elif self.strategy == "mvp":  # minimize variance
            return self._mvp_func(df_rets)
        elif self.strategy == "mdp":
            return self._mdp_func(df_rets)
        elif self.strategy == "rp":
            return self._rp_func(df_rets)
        elif self.strategy == "benchmark":
            return self._benchmark_func(df_rets)

    @staticmethod
    def get_port_vol(wgts: np.ndarray, cov_mat: np.ndarray) -> float:
        return np.sqrt(np.dot(wgts.T, np.dot(cov_mat, wgts)))

    @staticmethod
    def get_port_mu(wgts: np.ndarray, mu_vec: np.ndarray) -> float:
        return np.dot(wgts, mu_vec)

    @staticmethod
    def get_port_div_ratio(wgts: np.ndarray, cov_mat: np.ndarray) -> float:
        # weighted average volatility
        w_vol = np.dot(np.sqrt(np.diag(cov_mat)), wgts.T)
        # portfolio volatility
        p_vol = np.sqrt(np.dot(wgts.T, np.dot(cov_mat, wgts)))
        return w_vol / (p_vol + 1e-8)

    def get_cov_mat(self, df_rets: pd.DataFrame) -> np.ndarray:
        if self.cov_mat is None:
            if self.config is None or "cov" not in self.config or self.config["cov"] is None:
                self.cov_mat = df_rets.cov()
            else:
                self.cov_mat = self.config["cov"](df_rets, **self.config["cov_params"])
        return self.cov_mat

    def get_mu_vec(self, df_rets: pd.DataFrame):
        if self.mu_vec is None:
            self.mu_vec = df_rets.mean().values

    def set_cov_mat(self, cov_mat: pd.DataFrame):
        self.cov_mat = cov_mat

    def set_mu_vec(self, mu_vec: np.ndarray):
        self.mu_vec = mu_vec

    def get_port_txn_cost(self, wgts: np.ndarray, prev_wgts: np.ndarray):
        # return self.txn_cost * np.square(wgts - prev_wgts).sum()
        return self.txn_cost * np.abs(wgts - prev_wgts).sum()

    def get_marginal_risk_contribution(self, wgts: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
        # Function to calculate asset contribution to total risk
        sigma = self.get_port_vol(wgts, cov_mat)
        return np.multiply(np.dot(cov_mat, wgts), wgts.T) / (sigma ** 2)

    def get_rp_obj(self, wgts: np.ndarray, cov_mat: np.ndarray) -> float:
        sigma = self.get_port_vol(wgts, cov_mat)
        x = wgts / sigma
        return (np.dot(x.T, np.dot(cov_mat, x)) / 2.) - np.sum(np.log(x + 1e-10)) / cov_mat.shape[0]

    def _benchmark_func(self, df_rets: pd.DataFrame) -> dict:
        self.get_cov_mat(df_rets)
        symbols = df_rets.columns
        weigts = {s: 0 for s in symbols}
        weigts[self.config["benchmark"]] = 1.0   # allocate everything to the benchmark asset
        return weigts

    def _ew_func(self, df_rets: pd.DataFrame) -> dict:
        self.get_cov_mat(df_rets)
        symbols = df_rets.columns
        return {s: 1 / len(symbols) for s in symbols}

    def _ivw_func(self, df_rets: pd.DataFrame) -> dict:
        self.get_cov_mat(df_rets)
        inv_vars = 1.0 / df_rets.std(axis=0)
        return (inv_vars / inv_vars.sum()).to_dict()

    def _mr_func(self, df_rets: pd.DataFrame) -> dict:
        symbols = df_rets.columns
        n_asset = len(symbols)
        self.get_mu_vec(df_rets)
        self.get_cov_mat(df_rets)
        w = cp.Variable(n_asset)
        ret = self.mu_vec @ w
        # define constraints
        constraints = [
            cp.sum(w) == 1,
            w <= 1,
            w >= 0
        ]
        if self.prev_weights is not None:
            objective = cp.Maximize(ret - self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Maximize(ret)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)
        return {k: v for k, v in zip(symbols, set_eps_wgt_to_zeros(w.value))}

    def _mvp_func(self, df_rets: pd.DataFrame) -> dict:
        # maximum return portfolio
        symbols = df_rets.columns
        n_asset = len(symbols)
        self.get_mu_vec(df_rets)
        self.get_cov_mat(df_rets)
        w = cp.Variable(n_asset)
        # define constraints
        constraints = [
            cp.sum(w) == 1,
            w <= 1,
            w >= 0
        ]
        # compute risk
        risk = cp.quad_form(w, self.cov_mat)

        if self.prev_weights is not None:
            objective = cp.Minimize(risk + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)
        return {k: v for k, v in zip(symbols, set_eps_wgt_to_zeros(w.value))}

    def _mvo_func(self, df_rets: pd.DataFrame) -> dict:
        assert "vol_target" in self.config.keys(), "volatility target constraint must be provided!"
        assert self.config["vol_target"] > 0, "volatility target must be larger than 0."
        symbols = df_rets.columns
        n_asset = len(symbols)
        self.get_mu_vec(df_rets)
        self.get_cov_mat(df_rets)
        w = cp.Variable(n_asset)
        ret = self.mu_vec @ w
        # define constraints
        constraints = [
            cp.sum(w) == 1,
            w <= 1,
            w >= 0
        ]
        # compute risk
        risk = cp.quad_form(w, self.cov_mat)
        constraints += [risk <= (self.config["vol_target"] ** 2)]

        if self.prev_weights is not None:
            objective = cp.Maximize(ret - self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Maximize(ret)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)
        return {k: v for k, v in zip(symbols, set_eps_wgt_to_zeros(w.value))}

    def _mdp_func(self, df_rets: pd.DataFrame) -> dict:

        symbols = df_rets.columns
        n_asset = len(symbols)
        self.get_cov_mat(df_rets)
        if self.prev_weights is not None:
            cost = lambda wgts: -self.get_port_div_ratio(wgts, self.cov_mat) + \
                                self.get_port_txn_cost(wgts, self.prev_weights)
        else:
            cost = lambda wgts: -self.get_port_div_ratio(wgts, self.cov_mat)

        if self.init_weights is not None:
            init_weights = self.init_weights
        else:
            init_weights = np.array(n_asset * [1. / n_asset])

        opt = minimize(cost, x0=init_weights,
                       bounds=tuple((self.min_weight, self.max_weight) for k in range(n_asset)),
                       constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}],
                       method='SLSQP')
        return {k: v for k, v in zip(symbols, opt["x"])}

    def _rp_func(self, df_rets: pd.DataFrame) -> dict:
        symbols = df_rets.columns
        n_asset = len(symbols)
        self.get_cov_mat(df_rets)
        self.get_mu_vec(df_rets)
        w = cp.Variable(n_asset)

        # compute risk
        risk = cp.quad_form(w, self.cov_mat)

        # define constraints
        constraints = [
            cp.sum(cp.log(w)) >= 1.0 / n_asset,
            w >= 0
        ]

        if self.prev_weights is not None:
            objective = cp.Minimize(risk + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)
        w.value = w.value / np.sum(w.value)

        return {k: v for k, v in zip(symbols, set_eps_wgt_to_zeros(w.value))}
