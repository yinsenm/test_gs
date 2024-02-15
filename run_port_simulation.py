import os
import sys
sys.path.append("../src")
from lib_portfolio_evaluator import PortfolioEvaluator
from stcs import *
from glob import glob
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed


def process_stragegies_simu(file_names, output_name):
    pe = PortfolioEvaluator(begin_date=begin_date, end_date=end_date, df_monthly_rets=df_rets)
    dict_allocations = {}
    for file_name in file_names:
        strategy_name = os.path.basename(file_name).replace(".pickle", "")
        with open(file_name, "rb") as f:
            dict_allocations[strategy_name] = pickle.load(f)["allocations"]
    pe.set_portfolios(dict_allocations=dict_allocations)
    pe.compute_portfolios_values(
        cash_start=cash_start,
        transaction_cost=transaction_cost,
    )
    pe.evaluate_portfolios().astype(float).round(3).\
        to_csv("../results/%s/performance_%s_%s-%s-%dbps.csv" %
               (output_folder, output_name, begin_date.replace("-", ""), end_date.replace("-", ""), transaction_cost))


if __name__ == "__main__":
    file = "prcs_n9_1988_2023.csv"
    # evaluate performance of each strategy
    begin_date = "1990-01-31"
    end_date = "2020-12-31"
    transaction_cost = 10  # transaction cost in bps
    cash_start = 100000
    folder = "jam_n9_cvx"
    output_folder = "%s/csv/%s-%s" % (folder, begin_date.replace("-", ""), end_date.replace("-", ""))
    os.makedirs("../results/%s" % output_folder, exist_ok=True)
    df_prcs = pd.read_csv("../data/%s" % file, parse_dates=["Date"]).\
        set_index("Date").resample("M", label="right").last()
    df_rets = df_prcs.pct_change().dropna()
    n_jobs = 5  # parallel computing

    file_names = glob("../results/%s/*.pickle" % folder)
    df_files = pd.DataFrame.from_dict(
        {file_name: os.path.basename(file_name).replace(".pickle", "") for file_name in file_names},
        orient='index', columns=["strategy"]
    )
    df_files["target"] = df_files["strategy"].str.split("-").str[0]
    df_files["cov_func"] = df_files["strategy"].str.split("-").str[-1]
    df_files.sort_values(["cov_func", "strategy"], inplace=True)

    dict_of_strategies = {
        k: v for k, v in zip(df_files.groupby('cov_func').groups.keys(), df_files.groupby('cov_func').groups.values())
    }

    # for k, v in tqdm(dict_of_strategies.items()):
    #     process_stragegies_simu(v, k)
    # run portfolio simulation in parallel
    Parallel(n_jobs)(delayed(process_stragegies_simu)(v, k) for k, v in dict_of_strategies.items())

