import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm
from scipy import stats
from math import sqrt
import pandas_datareader.data as web # download 3m t-bill from FRED as risk-free rate
from glob import glob
import os
from tqdm import tqdm

class PortfolioEvaluator:
    def __init__(self,
                 begin_date: str, end_date: str,
                 df_monthly_rets: pd.DataFrame,
                 df_daily_rets: pd.DataFrame = None,
                 debug: bool = False):
        self.begin_date, self.end_date = begin_date, end_date
        self.symbols = df_monthly_rets.columns
        self.df_monthly_rets = df_monthly_rets.resample("M", label="right").last()
        self.df_daily_rets = df_daily_rets
        self.use_test_dates = self.df_monthly_rets[self.begin_date: self.end_date].index.tolist()

        self.dict_portfolios = {}    # dict to store portfolio, including cash position for vol-targeting strategy
        self.dict_allocations = {}  # dict to store asset allocation weights
        self.df_port_values = None # data frame to store the cumulative portfolio values
        self.vol_target = None
        self.transaction_cost = None
        self.df_rf = self.get_df_rf(begin_date=begin_date, end_date=end_date)  # get risk free rate
        self.debug = debug
        self.df_ports = None

    @staticmethod
    def get_df_rf(begin_date: str, end_date: str) -> pd.DataFrame:
        df_rf = web.DataReader("DGS3MO", "fred", begin_date, end_date)["DGS3MO"] / 100.
        df_rf = (1 + df_rf.resample("M", label="right").last()) ** (1 / 12.) - 1.0
        df_rf.index.name = "Date"
        return df_rf

    def load_portfolios(self, load_path: str):
        method_names = [os.path.basename(file).replace(".pickle", "") for file in glob("%s/*.pickle" % load_path)]
        for method_name in tqdm(sorted(method_names)):
            with open("%s/%s.pickle" % (load_path, method_name), "rb") as file:
                dict_method = pickle.load(file)
                self.dict_allocations[method_name] = dict_method["allocations"]

    def set_portfolios(self, dict_allocations):
        self.dict_allocations = dict_allocations

    @staticmethod
    def compute_txn_cost(old_weights: pd.Series, new_weights: pd.Series, transaction_cost: float) -> float:
        assert len(old_weights) == len(new_weights), "length of old_weights and new_weights must be equal"
        txn_sign = np.sign(old_weights - new_weights).round(5)
        x = (1 - transaction_cost * (txn_sign * old_weights).sum()) / \
            (1 - transaction_cost * (txn_sign * new_weights).sum())
        return 1 - x

    def compute_portfolios_values(self, cash_start: float, transaction_cost: float,
                                  vol_target: float = None, margin_finance_spread: float = 0,
                                  vol_long_day = 21, vol_short_day = 14):
        if vol_target is not None:
            assert self.df_daily_rets is not None, "df_daily_rets shall be provided for vol targeting strategy"
            assert vol_target > 0, "vol_target must be larger than 0"
            assert margin_finance_spread >= 0, "margin_finance_spread must be larger or equal to 0"
            self.vol_target = vol_target

        # self.transaction_cost = transaction_cost
        for method_name in list(self.dict_allocations.keys()):
            for ts_idx, ts_date in enumerate(self.use_test_dates):
                port_wgts = self.dict_allocations[method_name][ts_date]["weights"][self.symbols]
                if vol_target is not None:  # estimate historical volatility
                    rvs_vol = sqrt(252) * max(
                        (self.df_daily_rets.loc[: ts_date].iloc[-vol_long_day:] * port_wgts).sum(axis=1).std(),
                        (self.df_daily_rets.loc[: ts_date].iloc[-vol_short_day:] * port_wgts).sum(axis=1).std(),
                    )
                    port_wgts = port_wgts / rvs_vol  # scale the weights using the target volatility level
                    port_wgts["Cash"] = 1 - port_wgts.sum()  # add cash position
                else:
                    port_wgts["Cash"] = 0

                if ts_idx == 0:
                    self.dict_portfolios[method_name] = []
                    txn_cost = self.compute_txn_cost(old_weights=pd.Series(0, index=self.symbols.to_list() + ["Cash"]),
                                                     new_weights=port_wgts,
                                                     transaction_cost=transaction_cost / 10000.)
                    txn_cost_amount = cash_start * txn_cost
                    port_vals = port_wgts * (cash_start - txn_cost_amount)
                    port_wgts_plus = None
                    port_vals_plus = None
                    turnover = 0
                else:
                    port_rets = self.df_monthly_rets.loc[ts_date].fillna(0.)
                    rf = self.df_rf[: ts_date].iloc[-1]  # get the latest risk-free rate
                    # check the cash position is on margin or on deposit
                    if self.dict_portfolios[method_name][-1]["weights"]["Cash"] < 0:
                        # if cash on margin, the cost of borrow is rf + margin_finance_spread
                        port_rets["Cash"] = ((1 + rf) ** 12 + margin_finance_spread / 100) ** (1. / 12) - 1.0
                    else:
                        port_rets["Cash"] = ((1 + rf) ** 12) ** (1. / 12) - 1.0

                    # step 1, get asset appreciation before rebalancing
                    port_vals_plus = self.dict_portfolios[method_name][-1]["values"] * (1.0 + port_rets)
                    port_wgts_plus = port_vals_plus / port_vals_plus.sum()

                    # step 2: compute the cost for rebalancing
                    txn_cost = self.compute_txn_cost(
                        old_weights=port_wgts_plus,
                        new_weights=port_wgts,
                        transaction_cost=transaction_cost / 10000.
                    )
                    txn_cost_amount = port_vals_plus.sum() * txn_cost

                    # step 3: subtract the transaction cost and update new portfolio values
                    port_vals = port_wgts * port_vals_plus.sum() * (1 - txn_cost)
                    turnover = (port_wgts - port_wgts_plus).abs().sum()  # sum of absolute of change in weights

                # append new status of each portfolio
                self.dict_portfolios[method_name].append(
                    {
                        "Date": ts_date,
                        "weights": port_wgts,
                        "weights+": port_wgts_plus,
                        "turnover": turnover,
                        "values": port_vals,
                        "values+": port_vals_plus,
                        "txn_cost_amount": txn_cost_amount,
                        "portfolio_value": port_vals.sum()
                    }
                )

            if self.debug:
                print("%20s completed!" % method_name)

    def get_turnover_metrics(self) -> tuple:
        turnovers = []
        txn_cost_amounts = []
        for method_name in self.dict_portfolios.keys():
            n = len(self.dict_portfolios[method_name]) - 1
            turnovers.append(sum([_["turnover"] for _ in self.dict_portfolios[method_name]]) / n)
            txn_cost_amounts.append(sum([_["txn_cost_amount"] for _ in self.dict_portfolios[method_name]]))
        return turnovers, txn_cost_amounts

    def get_diversification_metrics(self) -> tuple:
        hhis = []
        effs = []
        for method_name in self.dict_portfolios.keys():
            n = len(self.dict_portfolios[method_name]) - 1
            hhis.append(sum([sum(_["weights"] ** 2) for _ in self.dict_portfolios[method_name][1:]]) / n)
            effs.append(1.0 / hhis[-1])
        return hhis, effs

    def get_metrics(self, prcs, rets, excess_rets):
        metrics = {}
        metrics["Sharpe Ratio"] = excess_rets.mean() / rets.std()
        metrics["Annualized Sharpe Ratio"] = metrics["Sharpe Ratio"] * sqrt(12)
        metrics["Skewness"] = rets.skew()
        metrics["Kurtosis"] = stats.kurtosis(rets.to_list(), fisher=False)
        metrics["Adjusted Sharpe Ratio"] = metrics["Sharpe Ratio"] * (
                1.0 + (metrics["Skewness"] / 6.0) * metrics["Sharpe Ratio"] -
                ((metrics["Kurtosis"] - 3) / 24.) * metrics["Sharpe Ratio"] ** 2
        )

        metrics["Annualized STD (%)"] = sqrt(12) * rets.std() * 100.
        metrics["Annualized Kurtosis"] = stats.kurtosis((12 * rets).to_list(), fisher=False)
        metrics["Annualized Skewness"] = (12 * rets).skew()
        metrics["Cumulative Return (%)"] = ((1. + rets).prod() - 1.0) * 100.
        metrics["Annual Return (%)"] = (1.0 + rets).groupby(rets.index.year).prod() - 1.0

        n = len(metrics["Annual Return (%)"])
        metrics["Arithmetic Return (%)"] = metrics["Annual Return (%)"].mean() * 100.
        metrics["Geometric Return (%)"] = (((metrics["Annual Return (%)"] + 1.).prod()) ** (1.0 / n) - 1.0) * 100.

        # compute VaR
        metrics["Monthly 95% VaR (%)"] = rets.quantile(0.05) * 100.
        metrics["Alt Monthly 95% VaR (%)"] = norm.ppf(0.05, rets.mean(), rets.std()) * 100.

        # compute maximum drawdown
        rolling_max = prcs.expanding().max()
        drawdown = prcs / rolling_max - 1.0
        metrics["Maximum Drawdown (%)"] = drawdown.min() * 100.
        metrics["MDD / VOL"] = metrics["Maximum Drawdown (%)"] / metrics["Annualized STD (%)"]

        dd = np.sqrt(np.sum(np.minimum(excess_rets, 0) ** 2) / len(excess_rets))
        metrics["Sortino Ratio"] = excess_rets.mean() / (dd + 1e-8)
        metrics["Annualized Sortino Ratio"] = sqrt(12) * metrics["Sortino Ratio"]

        metrics["Calmar Ratio"] = metrics["Geometric Return (%)"] / (metrics["Maximum Drawdown (%)"] + 1e-8)
        return metrics

    def get_port_return_metrics(self):
        ports = []
        for method_name in self.dict_portfolios.keys():
            df_port = pd.DataFrame(self.dict_portfolios[method_name])[["Date", "portfolio_value"]].set_index("Date")
            df_port.columns = [method_name]
            ports.append(df_port)
        self.df_ports = pd.concat(ports, axis=1).resample("M", label="right").last()

        port_ret_metrics = {}
        for port_name in self.df_ports.columns:
            prcs = self.df_ports[port_name]
            rets = prcs.pct_change().dropna()
            excess_rets = rets - self.df_rf[rets.index].mean()
            port_ret_metrics[port_name] = self.get_metrics(prcs=prcs, rets=rets, excess_rets=excess_rets)
        return port_ret_metrics

    def evaluate_portfolios(self, used_metrics=None):
        if used_metrics is None:
            used_metrics = [
                "Arithmetic Return (%)",
                "Geometric Return (%)",
                "Annualized STD (%)",
                "Cumulative Return (%)",
                "Maximum Drawdown (%)",
                "MDD / VOL",
                "Monthly 95% VaR (%)",
                "Sharpe Ratio",
                "Adjusted Sharpe Ratio",
                "Annualized Sharpe Ratio",
                "Annualized Sortino Ratio",
                "Calmar Ratio",
                "Turnover (%)",
                "Transaction Cost ($)",
                "Effective Holdings"
            ]

        txn_tos, txn_costs = self.get_turnover_metrics()
        port_hhis, port_effs = self.get_diversification_metrics()
        port_metrics = self.get_port_return_metrics()
        for idx, port_name in enumerate(port_metrics.keys()):
            port_metrics[port_name]["Turnover (%)"] = txn_tos[idx] * 100.
            port_metrics[port_name]["Transaction Cost ($)"] = txn_costs[idx]
            port_metrics[port_name]["HHI"] = port_hhis[idx]
            port_metrics[port_name]["Effective Holdings"] = port_effs[idx]

        df_performances = pd.DataFrame.from_dict(port_metrics)
        df_performances.columns = self.dict_portfolios.keys()
        return df_performances.loc[used_metrics]

