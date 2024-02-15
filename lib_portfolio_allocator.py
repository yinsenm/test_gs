import pickle
import os
import pandas as pd
from lib_portfolio_strategy import PortfolioStrategy
from tqdm import tqdm
from math import sqrt

class PortfolioAllocator:
    def __init__(self,
                 df_monthly_rets: pd.DataFrame,
                 begin_date: str = None,
                 end_date: str = None,
                 lookback_window: int = 24,
                 txn_cost: float = 10):
        self.df_monthly_rets = df_monthly_rets.resample("M", label="right").last()
        self.symbols = df_monthly_rets.columns
        assert lookback_window > 0, "the lookback window must larger than 0"
        self.begin_date = begin_date
        self.end_date = end_date
        if self.end_date is None:
            self.end_date = self.df_monthly_rets.index[-1]

        if self.begin_date is None:
            self.begin_date = self.df_monthly_rets.index[lookback_window]

        self.lookback_window = lookback_window
        self.txn_cost = txn_cost
        self.dates = self.df_monthly_rets.loc[self.begin_date: self.end_date].index
        self.dict_portfolios = {}
        self.dict_covs = {}

    def set_allocators(self, dict_portfolios_configs: dict):
        for name in dict_portfolios_configs:
            port_config = dict_portfolios_configs[name]
            self.dict_portfolios[name] = {}
            self.dict_portfolios[name]["strategy"] = port_config["strategy"]
            self.dict_portfolios[name]["config"] = port_config["config"] if "config" in port_config else None
            self.dict_portfolios[name]["allocator_cls"] = PortfolioStrategy(
                strategy=self.dict_portfolios[name]["strategy"],
                config=self.dict_portfolios[name]["config"],
                txn_cost=self.txn_cost
            )

        # save pre-computed covariance
        cov_names = set([cov_name.split("-")[1] for cov_name in dict_portfolios_configs.keys() if "-" in cov_name])
        for cov_name in cov_names:
            self.dict_covs[cov_name] = {}

    def allocate_weights(self, verbose: bool = True):
        for name in self.dict_portfolios:
            self.dict_portfolios[name]["allocations"] = {}

        prev_date = None
        for date in tqdm(self.dates, disable=not verbose):
            df_sub_rets = self.df_monthly_rets[:date].tail(self.lookback_window)

            init_weights = None
            for name in self.dict_portfolios:
                cov_name = name.split("-")[1] if "-" in name else None
                strat_cls = self.dict_portfolios[name]["allocator_cls"]

                # input the saved covariance matrix into strat_cls if we computed it before
                if cov_name is not None and date in self.dict_covs[cov_name]:
                    strat_cls.cov_mat = self.dict_covs[cov_name][date]
                else:
                    strat_cls.cov_mat = None

                # input previous weights to the optimizer to reduce turnover
                if prev_date is not None and self.txn_cost != 0:
                    prev_weights = self.dict_portfolios[name]["allocations"][prev_date]["weights"].copy()
                    # compute the updated weights after value appreciation
                    prev_weights *= (1 + df_sub_rets.iloc[-1])
                    prev_weights /= prev_weights.sum()
                    strat_cls.prev_weights = prev_weights.values

                # input initial weights to mvo to help convergence of the mvo algorithm
                if init_weights is not None and "mvo" in name:
                    strat_cls.init_weights = init_weights

                if "mvo" not in name:
                    weights = pd.Series(strat_cls.get_weights(df_sub_rets))
                elif "mvo" in name:
                    tgt_vol = sqrt(12) * self.dict_portfolios[name]["config"]["vol_target"]
                    if tgt_vol < mvo_min_vol:
                        weights = mvo_min_wgt
                    elif tgt_vol > mvo_max_vol:
                        weights = mvo_max_wgt
                    else:
                        weights = pd.Series(strat_cls.get_weights(df_sub_rets))

                # use previous mvp or mvo's (smaller vol) weights as the initial for next mvo (larger vol)
                if ("mvp" in name) or ("mvo" in name):
                    init_weights = weights.values
                else:
                    init_weights = None

                # update portfolio
                self.dict_portfolios[name]["allocations"][date] = {}
                self.dict_portfolios[name]["allocations"][date] = {
                    "weights": weights,
                    "port_vol":  sqrt(12) * strat_cls.get_port_vol(weights, strat_cls.cov_mat),   # annualized port vol
                    "port_div_ratio": strat_cls.get_port_div_ratio(weights, strat_cls.cov_mat),   # port diverification ratio
                    "port_mrc": strat_cls.get_marginal_risk_contribution(weights, strat_cls.cov_mat),
                }

                if "mvp" in name:
                    mvo_min_vol = sqrt(12) * strat_cls.get_port_vol(weights, strat_cls.cov_mat)
                    mvo_min_wgt = weights

                if "mr" in name:
                    mvo_max_vol = sqrt(12) * strat_cls.get_port_vol(weights, strat_cls.cov_mat)
                    mvo_max_wgt = weights

                # save covariance matrix
                if cov_name is not None:
                    self.dict_covs[cov_name][date] = strat_cls.cov_mat

            prev_date = date


    def save_portfolios(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs("%s/cov" % save_path, exist_ok=True)
        for name in self.dict_portfolios:
            with open("%s/%s.pickle" % (save_path, name), "wb") as file:
                pickle.dump(self.dict_portfolios[name], file)

        for cov_name in self.dict_covs:
            with open("%s/cov/%s.pickle" % (save_path, cov_name), "wb") as file:
                pickle.dump(self.dict_covs[cov_name], file)