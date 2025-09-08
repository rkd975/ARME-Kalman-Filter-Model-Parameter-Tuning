# ARME-Kalman-Filter-Model-Parameter-Tuning
- Code for optimising parameters for both artificial and real data (files beginning with RD_... were written/edited by me)
- Best ran in Google Colab (RD_optreal_results will **only run in Colab**) or in a Jupyter Notebook (with pandas library)
- All code in python

## !Warning! Running RD_optreal_results 
- **Must** run in Google Colab
- Need to **manually upload virtuoso.csv** into a prompt once you click *run* on the first cell block

## Code Contents
### RD_optimise:
- Runs optimisation for choice of parameters and choice of static/dynamic alpha
- Plots KF predicted s timeseries vs true s timeseries
- Plots heatmap over process and observation noise parameter space 

### RD_optart_results:
- Runs optimisation for 4 cases (static, dynamic) x (realistic input, ideal input)
- Can change number of iterations
- Plots KF and bGLS predicted s timeseries vs true timeseries
- Plots KF and bGLS predicted alpha timeseries vs true alpha timeseries
- Returns metrics about improvement in corr/std after optimisation
- Returns metrics about KF vs bGLS corr/std


### RD_optreal_results:
- Runs optimisation for all cases (leader, condition, repetition, loss type) and returns downloads a file 
- Plots profile loss plots for each parameter
- Plots heatmaps of std of asynchronies and lag-1 autocorrelation over noise parameter space
- Plots asynchrony timeseries
- Plots heatmaps of combined loss over noise parameter space for each condition (DP, NR, SP)
- (contains some hidden plots that have been commented out)

## PowerPoint Summary of Analysis
[Link to PowerPoint](./slides/ARME%20Analysis%20Summary.pptx)
