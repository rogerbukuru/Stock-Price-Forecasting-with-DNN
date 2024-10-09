# How to replicate plots

## AAR Plots and Per Share Contribution Plots
- In the `plot_results.ipy` simply change the `beta_value` variable e.g set it to 0.2 and the appropriate data file will be loaded and plotted.
- Similary for the per share contribution plots in the `individual_share_analysis.ipynb` simply change the `beta_value` and the appropriate data file will be loaded and plotted.


# How to execute the Short-Term Investment Network

In the `app.py` we have introduced a new variable called `investment_horizon` this variable takes the values `short` or `long`.
 - In order to execute the short-term investment Bayesian Decision Network (BDN) set this to `short` otherwise set it to `long` for the default system setup.