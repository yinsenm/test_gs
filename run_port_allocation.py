import sys
sys.path.append("../src")
from lib_portfolio_strategy_cvx import get_strats
from lib_portfolio_allocator import PortfolioAllocator
from lib_cov_func import *
from stcs import *
from joblib import Parallel, delayed
from tqdm import tqdm

file = "prcs_n9_1988_2023.csv"
folder = "jam_n9_cvx"
df_prcs = pd.read_csv("../data/%s" % file, parse_dates=["Date"]).set_index("Date").resample("M", label="right").last()
df_rets = df_prcs.pct_change().dropna()

# define target volatility
vol_targets = np.arange(3, 16, step=1)
list_of_strategies = list()
list_of_strategies.append({
    "ew": {"strategy": "ew"},
    "ivw": {"strategy": "ivw"},
    "spx": {"strategy": "benchmark", "config": {"benchmark": "SPX"}},
    "lbustruu": {"strategy": "benchmark", "config": {"benchmark": "LBUSTRUU"}},
})

# add hc
list_of_strategies.append(get_strats("hc", None, {}, vol_targets))

# add ledoit
list_of_strategies.append(get_strats("ledoit", ledoit, {}, vol_targets))

# # add shrinkage methods 1 to 8
# list_of_strategies.append(get_strats("LS1", cov1Para, {}, vol_targets))
# list_of_strategies.append(get_strats("LS2", cov2Para, {}, vol_targets))
# list_of_strategies.append(get_strats("LS3", covCor, {}, vol_targets))
# list_of_strategies.append(get_strats("LS4", covDiag, {}, vol_targets))
# list_of_strategies.append(get_strats("LS5", covMarket, {}, vol_targets))
# list_of_strategies.append(get_strats("NLS6", GIS, {}, vol_targets))
# list_of_strategies.append(get_strats("NLS7", LIS, {}, vol_targets))
# list_of_strategies.append(get_strats("NLS8", QIS, {}, vol_targets))

# add gs1
for ts in [.5, .7, .9]:
    list_of_strategies.append(
        get_strats("gs1=ts%.1f" % ts, gerber_cov_stat1, {"threshold": ts}, vol_targets)
    )

# add gs2
for ts in [.5, .7, .9]:
    list_of_strategies.append(
        get_strats("gs2=ts%.1f" % ts, gerber_cov_stat2, {"threshold": ts}, vol_targets)
    )

# add mgs
for gamma in [0.5, 1, 2, 3, 4, 5, 10]:
    list_of_strategies.append(
        get_strats("mgs=gam%.1f" % gamma, modified_gerber_cov, {"gamma": gamma}, vol_targets)
    )

# # add IQs
# df_iq_cov_configs = pd.read_csv("../config/collection_200_modified.csv").set_index("IQ")
# df_iq_cov_configs["dDminus"] = df_iq_cov_configs["dDplus"]
# for idx, row in df_iq_cov_configs.iterrows():
#     list_of_strategies.append(
#         get_strats("IQ=%03d" % idx, IQ, row.to_dict(), vol_targets)
#     )

print("Strategies generated success!")

# allocate portfolio based on each strategy
def process_stragegies(dict_of_strategies):
    pa = PortfolioAllocator(df_monthly_rets=df_rets)
    pa.set_allocators(dict_portfolios_configs=dict_of_strategies)
    pa.allocate_weights(verbose=False)
    pa.save_portfolios(save_path="../results/%s" % folder)

if __name__ == "__main__":
    Parallel(n_jobs=4)(
        delayed(process_stragegies)(dict_of_strategies) for dict_of_strategies in tqdm(list_of_strategies)
    )

