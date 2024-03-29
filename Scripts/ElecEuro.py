# ElecEuro.py: various helpers for running regressions on entsoe data
# author: Ethan Reese
# March 4, 2022

import os
import pandas as pd
import numpy as np
import datetime as dt
import emd
from sklearn import linear_model
import plotly.express as px
import multiprocessing
from functools import partial
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sklearn

from plotly.subplots import make_subplots
import plotly.graph_objects as go

""" Read and clean information about generator breakdowns

Keyword Arguments:
country_name: name of country under question
data_path: top directory containing all entsoe data

Returns: Pandas dataframe with information about all gen types in country
"""
def gen_read_clean(country_name, data_path = "/Users/Ethan/Dev/Thesis Work/Data"):
        init_directory = os.getcwd()

        os.chdir(os.path.join(data_path, country_name, "Generation"))

        gen_data = []
        for file in os.listdir():
                gen_data.append(pd.read_csv(file))
        
        gen_frame = pd.concat(gen_data)

        gen_frame = gen_frame.dropna(subset=["Fossil Gas  - Actual Aggregated [MW]"])
        gen_frame = gen_frame[gen_frame["Fossil Gas  - Actual Aggregated [MW]"] != "n/e"]
        gen_frame = gen_frame[gen_frame["Fossil Gas  - Actual Aggregated [MW]"] != "N/A"]
        gen_frame = gen_frame[gen_frame["Fossil Gas  - Actual Aggregated [MW]"] != "-"]
        gen_frame = gen_frame.replace("n/e", 0)
        gen_frame = gen_frame.drop("Area", axis=1)

        gen_frame["Start"] = gen_frame["MTU"].apply(lambda x: x.split("-")[0])
        gen_frame["Start"] = pd.to_datetime(gen_frame["Start"])
        gen_frame = gen_frame.drop(["MTU","Hydro Pumped Storage  - Actual Aggregated [MW]"], axis=1)

        gen_frame = gen_frame.set_index("Start")
        
        gen_frame = gen_frame.apply(pd.to_numeric)
        gen_frame["Fossil Gas  - Actual Aggregated [MW]"] = gen_frame["Fossil Gas  - Actual Aggregated [MW]"].apply(float)

        gen_frame["Total"] = gen_frame.sum(axis=1)
        gen_frame["Total"] = gen_frame["Total"].apply(float)
        gen_frame["Percent Gas"] = gen_frame["Fossil Gas  - Actual Aggregated [MW]"].divide(gen_frame["Total"])

        os.chdir(init_directory)

        return gen_frame

""" Combine dataframes of gas prices and generation types

Keyword Arguments:
country_name: name of country under question
gen_df: Pandas dataframe with generator breakdowns
data_path: top directory containing all entsoe data
freq: aggregation frequency for plotting purposes

returns dataframe with combined data and aggregations under period in frequency
"""
def combine_gen_gas(country_name, gen_df, data_directory="/Users/Ethan/Dev/Thesis Work/Data", freq="1D"):
        init_directory = os.getcwd()

        gas_data = pd.read_csv(os.path.join(data_directory, country_name, "combined_data.csv"))
        gas_data["Start"] = pd.to_datetime(gas_data["Start"])
        gas_data = gas_data.set_index("Start")
        gas_data = gas_data.groupby(pd.Grouper(freq=freq)).mean()

        daily_gen_avg = gen_df.groupby(pd.Grouper(freq=freq)).mean()
        gen_trimmed = daily_gen_avg[["Percent Gas", "Total", "Fossil Gas  - Actual Aggregated [MW]"]]
        
        combined_data = gas_data.merge(gen_trimmed, on = "Start", how = "left")
        
        os.chdir(init_directory)

        return combined_data.reset_index()

""" Read in and clean data for the electricity and LNG pricing

Keyword Arguments:
country: name of the country under question
lng_path: path to the lng data files
area_suffix: entso-e area suffix for column name cleaning
coal_path: path to Rotterdam coal data from Bloomberg
data_path: path to the top data file in the 

returns a combined, cleaned directory with commodity and elec prices
also writes this file to the country's data directory
"""
def read_and_clean(country, lng_path, area_suffix, coal_path = "/Users/Ethan/Dev/Thesis Work/Data/European Coal.csv",
                data_path = "/Users/Ethan/Dev/Thesis Work/Data"):
        init_directory = os.getcwd()
        # grab the capacity data to get started
        capacities = pd.read_csv(os.path.join(data_path, country, "capacity.csv"))
        cap_t = capacities.transpose()
        cap_t.columns = cap_t.iloc[0]
        cap_t["temp"] = cap_t.index
        cap_t["Year"] = cap_t["temp"].apply(lambda x: x[0:4])
        cap_t = cap_t.replace(to_replace=r'[^0-9.-]+', value=0, regex=True)
        cap_t = cap_t.fillna(0)
        cap_t = cap_t.reset_index()
        cap_t = cap_t.drop(0)

        # put all the capacity data together
        total_caps = pd.DataFrame()
        total_caps["Year"] = cap_t["Year"].astype(int)
        total_caps["Coal Capacity"] = cap_t["Fossil Hard coal"].astype(int) + cap_t["Fossil Brown coal/Lignite"].astype(int)
        total_caps["LNG Capacity"] = cap_t["Fossil Gas"]
        total_caps["Dispatch Renewable Capacity"] = cap_t["Hydro Run-of-river and poundage"].astype(int) + \
                cap_t["Nuclear"].astype(int) + cap_t["Hydro Water Reservoir"].astype(int)

        total_caps["Wind Onshore Total"] = cap_t["Wind Onshore"].astype(int)
        total_caps["Solar Total"] = cap_t["Solar"].astype(int)

        # start by iterating and reading in all of the files
        price_data = []

        os.chdir(os.path.join(data_path, country, "Price"))
        for file in os.listdir():
                price_data.append(pd.read_csv(file))

        prices = pd.concat(price_data)
        prices.to_csv("./prices.csv")

        # general clean up
        prices = prices.dropna(subset=["Day-ahead Price [EUR/MWh]"])
        prices = prices[prices["Day-ahead Price [EUR/MWh]"] != "n/e"]
        prices = prices[prices["Day-ahead Price [EUR/MWh]"] != "-"]
        prices["Price"] = prices["Day-ahead Price [EUR/MWh]"].apply(float)
        prices = prices[["MTU (CET/CEST)","Price"]]

        # convert the time range into a start + end time
        prices["Start"] = prices["MTU (CET/CEST)"].apply(lambda x: x.split("-")[0])
        prices["End"] = prices["MTU (CET/CEST)"].apply(lambda x: x.split("-")[1])


        ## READ AND CLEAN LOAD DATA
        # work through the same process for the load data
        load_data = []
        os.chdir(os.path.join("..", "Load"))
        for load_file in os.listdir():
                df = pd.read_csv(load_file)
                load_data.append(df)
        
        loads = pd.concat(load_data)

        # special step for Germany due to some quirks of reporting
        if(country == "Germany"):
                loads["Forecast"] = loads["Day-ahead Total Load Forecast [MW] - BZN|DE-AT-LU"].combine_first(loads["Day-ahead Total Load Forecast [MW] - BZN|DE-LU"])
                loads["Actual"] = loads["Actual Total Load [MW] - BZN|DE-AT-LU"].combine_first(loads["Actual Total Load [MW] - BZN|DE-LU"])
        elif(country == "Austria"):
                loads["Forecast"] = loads["Day-ahead Total Load Forecast [MW] - BZN|DE-AT-LU"].combine_first(loads["Day-ahead Total Load Forecast [MW] - BZN|AT"])
                loads["Actual"] = loads["Actual Total Load [MW] - BZN|DE-AT-LU"].combine_first(loads["Actual Total Load [MW] - BZN|AT"])
        else:
                loads["Forecast"] = loads["Day-ahead Total Load Forecast [MW] - {}".format(area_suffix)]
                loads["Actual"] = loads["Actual Total Load [MW] - {}".format(area_suffix)]
        loads = loads.dropna(subset=["Forecast", "Actual"])
        loads = loads[loads["Actual"] != "n/e"]
        loads = loads[loads["Actual"] != "-"]
        loads["Actual"] = loads["Actual"].apply(float)
        loads["Forecast"] = loads["Forecast"].apply(float)

        loads = loads[["Time (CET/CEST)", "Forecast", "Actual"]]
        # make the time range effective
        loads["Start"] = loads["Time (CET/CEST)"].apply(lambda x: x.split("-")[0])
        loads["End"] = loads["Time (CET/CEST)"].apply(lambda x: x.split("-")[1])

        ## READ AND CLEAN FORECAST DATASETS
        forecast_data = []
        os.chdir(os.path.join("..", "Forecasts"))
        for forecast_file in os.listdir():
                df = pd.read_csv(forecast_file)
                forecast_data.append(df)

        forecasts = pd.concat(forecast_data)
        forecasts_clean = pd.DataFrame()
        forecasts_clean["Start"] = forecasts["MTU (CET/CEST)"].apply(lambda x: x.split("-")[0])
        forecasts = forecasts.fillna(0)
        forecasts = forecasts.replace(to_replace="-", value=0)
        forecasts = forecasts.replace(to_replace=r'[^0-9.-]+', value=0, regex=True)

        forecasts_clean["Total Ren"] = forecasts.iloc[:, 1].astype(int) + forecasts.iloc[:, 2].astype(int) \
                + forecasts.iloc[:, 3].astype(int)

        os.chdir(os.path.join("..",".."))

        ## MERGING THE TWO DATASETS
        # merge the two sets of data
        elec_merged = pd.merge(left = prices, right = loads, on = "Start", how="left")
        elec_merged = elec_merged[["Start", "Price", "Actual", "Forecast"]]
        elec_merged = elec_merged.merge(forecasts_clean, on = "Start", how="left")
        elec_merged["Date"] = pd.to_datetime(pd.to_datetime(elec_merged["Start"]).dt.date)
        ## READ IN LNG DATA FROM BLOOMBERG
        lng_data = pd.read_excel(lng_path)
        # filter columns
        lng_data = lng_data[["Date", "Last Price"]]

        lng_data["Date"] = lng_data["Date"] - dt.timedelta(days=1)

        ## READ IN COAL DATA FROM BLOOMBERG NEF
        coal_data = pd.read_csv(coal_path)
        coal_data["Coal Price"] = coal_data["Time Series: Price"]
        # there are some recent missing days, so fill in those values with interpolation
        coal_data["Coal Price"] = coal_data["Coal Price"].interpolate()
        coal_data = coal_data.rename(columns={"Timeseries: Date Axis": "Date"})

        coal_data["Date"] = pd.to_datetime(coal_data["Date"])

        
        coal_data = coal_data[["Date", "Coal Price"]]

        combined = elec_merged.merge(lng_data, on="Date")
        combined = combined.merge(coal_data, on = "Date")

        combined = combined.dropna(subset=["Forecast", "Actual"])
        
        combined = combined.merge(coal_data, on="Date", how="left")

        combined["Year"] = combined["Date"].dt.year

        combined = combined.merge(total_caps, how = "left", on = "Year")

        combined.to_csv("./{}/combined_data.csv".format(country))

        os.chdir(init_directory)

        return combined

""" Applies EMD filter to the combined set of data and returns regression results

Keyword Arguments:
combined_data: combined dataframe with gas and electricity data
country: country the regression is performed on
low_pass_percent: the percent of the filters to filter out for low
med_pass_percent: the percent of the filters to filter out for med
high_pass_percent: the percent of the filters to filter out for high
log_adj: the term added to all log values to prevent negative values
verbose: whether to print values to console

returns a linear model fit on the given data
"""
def filter_and_regress(combined_data, country, 
        low_pass_percent=0.05, med_pass_percent=0.5, high_pass_percent=0.5, log_adj=20,
        verbose = True):
        
        combined_data = combined_data[["Last Price", "Price", "Forecast", "Coal Price_x"]]
        combined_data = combined_data.rename(columns={"Coal Price_x": "Coal Price"})
        combined_data = combined_data.dropna(axis=0)
        elec_price = combined_data["Price"].to_numpy()
        lng_price = combined_data["Last Price"].to_numpy()
        coal_price = combined_data["Coal Price"].to_numpy()
        demand = combined_data["Forecast"].to_numpy()
        # plot and transform all of the data for electricity pricing
        #plt.figure()
        #plt.plot(elec_price, "k")

        imf, noise = emd.sift.complete_ensemble_sift(elec_price, ensemble_noise=1)
        # create the pass thresholds based on the input percentages
        low_pass_thresh_elec = int(np.ceil(low_pass_percent * imf.shape[1]))
        med_pass_thresh_elec = int(np.ceil(med_pass_percent * imf.shape[1]))
        high_pass_thresh_elec = int(np.ceil(high_pass_percent * imf.shape[1]))
        #emd.plotting.plot_imfs(imf)

        IP, IF, IA = emd.spectra.frequency_transform(imf, 2156, "hilbert")
        # plot and transform all of the data for LNG prices
        #plt.figure()
        #plt.plot(lng_price, "k")

        lng_imf, lng_noise = emd.sift.complete_ensemble_sift(lng_price, ensemble_noise=1)
        low_pass_thresh_lng = int(np.ceil(low_pass_percent * lng_imf.shape[1]))
        med_pass_thresh_lng = int(np.ceil(med_pass_percent * lng_imf.shape[1]))
        high_pass_thresh_lng = int(np.ceil(high_pass_percent * lng_imf.shape[1]))

        #emd.plotting.plot_imfs(lng_imf)

        IP, IF, IA = emd.spectra.frequency_transform(imf, 2156, "hilbert")

        demand_imf, demand_noise = emd.sift.complete_ensemble_sift(demand, ensemble_noise=1)
        low_pass_thresh_demand = int(np.ceil(low_pass_percent * demand_imf.shape[1]))
        med_pass_thresh_demand = int(np.ceil(med_pass_percent * demand_imf.shape[1]))
        high_pass_thresh_demand = int(np.ceil(high_pass_percent * demand_imf.shape[1]))

        coal_imf, coal_noise = emd.sift.complete_ensemble_sift(coal_price, ensemble_noise=1)
        low_pass_thresh_coal = int(np.ceil(low_pass_percent * coal_imf.shape[1]))
        
        low_pass_elec = imf[:, low_pass_thresh_elec:]
        low_pass_means_elec = np.apply_along_axis(np.mean, 1, low_pass_elec)

        low_pass_lng = lng_imf[:, low_pass_thresh_lng:]
        low_pass_means_lng = np.apply_along_axis(np.mean, 1, low_pass_lng)

        low_pass_demand = demand_imf[:, low_pass_thresh_demand:]
        low_pass_means_demand = np.apply_along_axis(np.mean, 1, low_pass_demand)

        low_pass_coal = coal_imf[:, low_pass_thresh_coal:]
        low_pass_means_coal = np.apply_along_axis(np.mean, 1, low_pass_coal)

        #px.scatter(x=low_pass_means_elec, y=low_pass_means_lng)

        """
        med_pass_elec = imf[:, med_pass_thresh_elec:]
        med_pass_means_elec = np.apply_along_axis(np.mean, 1, med_pass_elec)

        med_pass_lng = lng_imf[:, med_pass_thresh_lng:]
        med_pass_means_lng = np.apply_along_axis(np.mean, 1, med_pass_lng)

        med_pass_demand = demand_imf[:, med_pass_thresh_demand:]
        med_pass_means_demand = np.apply_along_axis(np.mean, 1, med_pass_demand)
        """

        #px.scatter(x=med_pass_means_elec, y=med_pass_means_lng)
        """
        high_pass_elec = imf[:, high_pass_thresh_elec:]
        high_pass_means_elec = np.apply_along_axis(np.mean, 1, high_pass_elec)

        high_pass_lng = lng_imf[:, high_pass_thresh_lng:]
        high_pass_means_lng = np.apply_along_axis(np.mean, 1, high_pass_lng)

        high_pass_demand = lng_imf[:, high_pass_thresh_demand:]
        high_pass_means_demand = np.apply_along_axis(np.mean, 1, high_pass_demand)
        """
        #px.scatter(x=high_pass_means_elec, y=high_pass_means_lng)
        X_low = pd.DataFrame({"Log LNG": low_pass_means_lng, "Log Coal": low_pass_means_coal, "Forecast": low_pass_means_demand})
        X_low_log = X_low.copy()
        X_low_log["Log LNG"] = X_low_log["Log LNG"].apply(lambda x: np.log(x+log_adj))
        X_low_log["Log Coal"] = X_low_log["Log Coal"].apply(lambda x: np.log(x + log_adj))
        low_model = linear_model.LinearRegression().fit(X_low_log, np.log(low_pass_means_elec+log_adj))
        
        """
        X_med = pd.DataFrame({"Log LNG": med_pass_means_lng, "Forecast": med_pass_means_demand})
        X_med_log = X_med.copy()
        X_med_log["Log LNG"] = X_med_log["Log LNG"].apply(lambda x: np.log(x+log_adj))
        med_model = linear_model.LinearRegression().fit(X_med_log, np.log(med_pass_means_elec + log_adj))
        """
        if(verbose):
                print("Low thresh LNG coefficient = {}, Coal Coefficient = {}, Demand Coefficient = {}".format(low_model.coef_[0], low_model.coef_[1], low_model.coef_[1]))
                #print("Med thresh LNG coefficient = {}, Demand Coefficient = {}".format(med_model.coef_[0], med_model.coef_[1]))
        """
        X_high = pd.DataFrame({"LNG": high_pass_means_lng, "Demand": high_pass_means_demand})
        X_high_log = X_high.copy()
        X_high_log["LNG"] = X_high_log["LNG"].apply(lambda x: np.log(x + log_adj))
        high_model = linear_model.LinearRegression().fit(X_high_log, np.log(high_pass_means_elec + log_adj))
        print("High thresh LNG coefficient = {},  Demand Coefficent = {}".format(high_model.coef_[0], high_model.coef_[1]))
        """
        return low_model

def unfiltered_regression(combined_data, country, log_adj, time_start=7, time_end=11):
        combined_data = combined_data.dropna(axis=0)

        combined_data["Start"] = pd.to_datetime(combined_data["Start"])


        X, actual = get_X(combined_data, log_adj, time_start= time_start, time_end=time_end)

        model = linear_model.LinearRegression().fit(X, actual)

        return model

def get_X_aid(dataframe, log_adj, time_start=0, time_end=23):
        dataframe["Start"] = pd.to_datetime(dataframe["Start"])

        dataframe = dataframe[dataframe["Start"].dt.hour >= time_start]
        dataframe = dataframe[dataframe["Start"].dt.hour <= time_end]

        X = pd.DataFrame()
        X["Log LNG"] = dataframe["Last Price"]
        X["Log Coal"] = dataframe["Coal Price_x"]
        X["Forecast"] = dataframe["Forecast"]
        X["Capacity"] = (dataframe["Coal Capacity"] + dataframe["LNG Capacity"] + dataframe["Total Ren"] +\
        dataframe["Dispatch Renewable Capacity"])
        X["LNG Employed"] = ((dataframe["Forecast"]) - (dataframe["Total Ren"] + dataframe["Dispatch Renewable Capacity"] + dataframe["Coal Capacity"]) > 0).astype(int)
        X["Coal Employed"] = ((dataframe["Forecast"] - (dataframe["Total Ren"] + dataframe["Dispatch Renewable Capacity"])) > 0).astype(int) - X["LNG Employed"]
        X["Const"] = ((dataframe["Forecast"] - (dataframe["Total Ren"] + dataframe["Dispatch Renewable Capacity"])) < 0).astype(int)

        # properly break into the daily elements based on the period of the day considered.

        actual = dataframe["Price"]

        timeseries = dataframe["Start"]
        
        return X, actual, timeseries

def model_function(X, gamma, nu, lng_beta, coal_beta, ren_beta):
        # convention for maximum price of electricity
        M= 3000
        # C^max - D
        scarcity = X[:, 3]-X[:, 2]
        val = np.piecewise(scarcity, [scarcity > 0, scarcity <= 0],
                     [lambda scarcity: np.clip(np.divide(gamma,(np.power(scarcity, nu)), out = np.ones_like(scarcity) * M, where=(np.power(scarcity, nu))>0.01), a_min = 0, a_max=M),
                        M])
        return val * (lng_beta*X[:, 0]*X[:, 4]  + coal_beta*X[:, 1]*X[:, 5] + ren_beta*X[:,6])



def aid_regression(combined_data, country, log_adj, time_start=7, time_end=11, graphs = True, verbose = "a"):
        output_str = ""
        time_output = ""
        COVID_START = "2020-03-01"
        COVID_END = "2021-06-01"
        WAR_START = "2022-02-01"
        log_adj = max((-1 * min(np.min(combined_data["Price"]), np.min(combined_data["Last Price"]))) + 2, log_adj)
        combined_data = combined_data.dropna(axis=0)

        combined_data["Start"] = pd.to_datetime(combined_data["Start"])
        combined_data = combined_data.sort_values(by="Start")


        X, actual, ts = get_X_aid(combined_data, log_adj, time_start=time_start, time_end = time_end)
        popt, pcov = curve_fit(model_function, X.values, actual.values, p0=[10, 1, 1, 1, 1], maxfev=5000)
        output_str += ", ".join(str(i) for i in popt)
        output_str += "\n"

        time_output += output_str
        min_price = np.sum(actual < 0)
                #print(pcov)

        scarcity = X["Capacity"].values - X["Forecast"].values
        val = np.piecewise(scarcity, [scarcity > 0, scarcity <= 0],
                     [lambda scarcity: np.clip(popt[0]/(np.power(scarcity, popt[1])), a_min = 0, a_max=3000),
                        3000])
        
        output_str += "Number of Days: {} \n".format(len(actual))
        output_str += "g: {} \n".format(np.mean(val))
        df = pd.DataFrame({"Timeseries": ts, "Actual": actual, "Pred": model_function(X.values, *popt)})
        df["Timeseries"] = pd.to_datetime(df["Timeseries"])
        output_str += "{} R^2: {} \n".format(country, sklearn.metrics.r2_score(df["Actual"], df["Pred"]))
        if verbose == "time":
                pre = df.loc[df["Timeseries"] < COVID_START].copy()
                print(pre)
                
                covid = df.loc[df["Timeseries"] > COVID_START].copy()
                covid = covid.loc[covid["Timeseries"] < COVID_END].copy()

                war = df[df["Timeseries"] > WAR_START].copy()

                time_output += "{} Pre-Covid R^2: {} \n".format(country, sklearn.metrics.r2_score(pre["Actual"], pre["Pred"]))
                time_output += "{} Covid R^2: {} \n".format(country, sklearn.metrics.r2_score(covid["Actual"], covid["Pred"]))
                time_output += "{} War R^2: {} \n".format(country, sklearn.metrics.r2_score(war["Actual"], war["Pred"]))


                print(time_output)
        df.set_index("Timeseries", inplace=True)
        df = df.dropna(axis=0)
        df = df.resample("W").mean()
        df.reset_index(inplace=True)
        if graphs:
                #fig = px.line(df, x = "Timeseries", y= "Actual", title = country, name="Actual")
                fig = px.line(title=country)
                fig.add_scatter(x = df["Timeseries"], y= df["Actual"], name="Actual")
                fig.add_scatter(x=df["Timeseries"], y=df["Pred"], name = "Predicted")
                #fig.update_yaxes(type="log")
                fig.update_yaxes(title = "Marginal Price (€/MWh)")
                fig.update_xaxes(title = "Date")
                fig.update_layout(legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                        ))
                fig.show()
        df = df.dropna(axis=0)
        output_str += "{} R^2: {} \n".format(country, sklearn.metrics.r2_score(df["Actual"], df["Pred"]))
        if verbose == "a": 
                print(output_str)


        return output_str, df



def carmona_regression(combined_data, country, log_adj, time_start, time_end, graphs=False):
        log_adj = max((-1 * min(np.min(combined_data["Price"]), np.min(combined_data["Last Price"]))) + 2, log_adj)
        combined_data = combined_data.dropna(axis=0)

        combined_data["Start"] = pd.to_datetime(combined_data["Start"])
        combined_data = combined_data.sort_values(by="Start")


""" Produce graphs based on the coefficients calculated by the regression

Keyword arguments:
coefs_pre: coefficients for precov period
coefs_covid: coefficients for covid period
coefs_war: coefficients for the war period
country_name: name of the country under question for title purposes
combined_data: set of data to be used for the calculations
log_adj: the amount added to all values to make logs positive
group_freq: the frequency of aggregation on the graphing for clarity purposes
"""
# TODO: Update for coal
def produce_graphs(coefs_pre, coefs_covid, coefs_war, country_name, input_data, log_adj, group_freq = "1W"):
        # calculate the columns based on each coefficient
        combined_data = input_data[["Start", "Last Price", "Price", "Forecast"]].copy()

        
        combined_data["Log LNG"] = np.log(combined_data["Last Price"]+log_adj)
        X = combined_data[["Log LNG", "Forecast"]]

        combined_data["Pre Covid Log Prediction"] = coefs_pre.predict(X)
        combined_data["Pre Covid Prediction"] = combined_data["Pre Covid Log Prediction"].apply(np.exp)

        combined_data["During COVID Log Prediction"] = coefs_covid.predict(X)
        combined_data["During COVID Prediction"] = combined_data["During COVID Log Prediction"].apply(np.exp)

        combined_data["During War Log Prediction"] = coefs_war.predict(X)
        combined_data["During War Prediction"] = combined_data["During War Log Prediction"].apply(np.exp)
        
        combined_data["Start"] = pd.to_datetime(combined_data["Start"])
        combined_data = combined_data.set_index("Start")


        period_avg = combined_data.groupby(pd.Grouper(freq=group_freq)).mean()
        fig = px.line(period_avg, x = period_avg.index, y=period_avg["Pre Covid Prediction"], title = 
                        "{} Actual vs. Predicted Electricity Cost Over Time".format(country_name))

        fig.add_scatter(x = period_avg.index, y=period_avg["During War Prediction"], name = "During War Prediction")

        #fig.add_scatter(x = period_avg.index, y=period_avg["During COVID Prediction"], name = "During COVID Prediction")

        fig.add_scatter(x = period_avg.index, y=period_avg["Price"] + log_adj, name="Actual")

        fig.show()
        

""" Primary user facing function to run the regressions over various timeseries

Keyword Arguments:
combined_data_path: path to the combined data frame in filesystem
country_name: country under question for labelling
log_adj: the minimum adjustment for all log values
COVID_START: Start date for the COVID-19 Pandemic
COVID_END: End date for the COVID-19 Pandemic period
WAR_START: Start of war in Ukraine era

Shows a graph of the data with a fit line and prints information about the fit
"""
def timeperiod_differences(country_name, log_adj=1, COVID_START = "2020-03-01",
        COVID_END = "2021-06-01", WAR_START = "2022-02-01"):
        # these serve as best guesses, change at will
        # 50% of Europe was vaccinated by this date: 
        # https://www.bbc.com/news/explainers-52380823
        
        combined_data_path = os.path.join("/Users", "Ethan", "Dev", "Thesis Work", "Data", country_name, "combined_data.csv")

        time_blocks = {2:3, 7:8, 12:13, 18:19, 21:22}
        # read in the data from the combined dataset
        data = pd.read_csv(combined_data_path)

        # make the protocol for adjusting log values
        log_adj = max((-1 * min(np.min(data["Price"]), np.min(data["Last Price"]))) + 1, log_adj)

        
        data["Date"] = pd.to_datetime(data["Date"])
        pre_covid = data.loc[data["Date"] < COVID_START].copy()


        covid = data.loc[data["Date"] > COVID_START].copy()
        covid = covid.loc[covid["Date"] < COVID_END].copy()

        war = data[data["Date"] > WAR_START].copy()

        # run the regressions on the given datasets
        

        
        for start in time_blocks.keys():
                print("Coefficients for {} to {}".format(start, time_blocks[start]))
                model_pre = unfiltered_regression(pre_covid, country_name, log_adj, time_start=start, time_end= time_blocks[start])
                X_pre, actual_pre = get_X(pre_covid, log_adj, time_start=start, time_end=time_blocks[start])

                print("Pre-COVID in {}".format(country_name))
                print("LNG Beta: {}, Coal Beta: {}, Demand Beta: {}".format(model_pre.coef_[0], model_pre.coef_[1], model_pre.coef_[2]))
                print("R^2: {}".format(model_pre.score(X_pre, actual_pre)))

                model_cov = unfiltered_regression(covid, country_name, log_adj, time_start=start, time_end= time_blocks[start])
                X_cov, actual_cov = get_X(covid, log_adj, time_start=start, time_end=time_blocks[start])

                print("COVID Era in {}".format(country_name))
                print("LNG Beta: {}, Coal Beta: {}, Demand Beta: {}".format(model_cov.coef_[0], model_cov.coef_[1], model_cov.coef_[2]))
                print("R^2: {}".format(model_cov.score(X_cov, actual_cov)))

                model_war = unfiltered_regression(war, country_name, log_adj, time_start=start, time_end= time_blocks[start])
                X_war, actual_war = get_X(war, log_adj, time_start=start, time_end=time_blocks[start])

                print("War Era in {}".format(country_name))
                print("LNG Beta: {}, Coal Beta: {}, Demand Beta: {}".format(model_war.coef_[0], model_war.coef_[1], model_war.coef_[2]))
                print("R^2: {}".format(model_war.score(X_war, actual_war)))
                print("")

        """
        pre_covid["Log LNG"] = np.log(pre_covid["Last Price"] + log_adj)
        X_pre = pre_covid[["Log LNG", "Forecast"]]
        pre_covid["Log Elec"] = np.log(pre_covid["Price"] + log_adj)
        print("R^2: {}".format(coefs_pre.score(X_pre, pre_covid["Price"])))

        print("COVID Era in {}".format(country_name))
        coefs_covid = filter_and_regress(covid, country_name, log_adj=log_adj)

        covid["Log LNG"] = np.log(covid["Last Price"]+log_adj)

        X_cov = covid[["Log LNG", "Forecast"]]
        covid["Log Elec"] = np.log(covid["Price"]+log_adj)
        print("R^2: {}".format(coefs_pre.score(X_cov, covid["Price"])))

        print("War in Ukraine Era in {}".format(country_name))
        coefs_war = filter_and_regress(war, country_name, log_adj=log_adj)

        war["Log LNG"] = np.log(war["Last Price"]+log_adj)
        X_war = war[["Log LNG", "Forecast"]]
        war["Log Elec"] = np.log(war["Price"]+log_adj)
        print("R^2: {}".format(coefs_pre.score(X_war, war["Price"])))

        if (country_name == "Germany"):
                data = data[(data["Date"] < "2022-01-01") | (data["Date"] > "2022-03-01")]

        produce_graphs(coefs_pre, coefs_covid, coefs_war, country_name, data, log_adj)
        """
def aid_timeperiod_differences(country_name, log_adj=1, COVID_START = "2020-03-01",
        COVID_END = "2021-06-01", WAR_START = "2021-06-02"):
        # these serve as best guesses, change at will
        # 50% of Europe was vaccinated by this date: 
        # https://www.bbc.com/news/explainers-52380823
        
        combined_data_path = os.path.join("/Users", "Ethan", "Dev", "Thesis Work", "Data", country_name, "combined_data.csv")

        # time_blocks = {2:3, 8:9, 12:13, 18:19}
        time_blocks = {18:19}
        # read in the data from the combined dataset
        data = pd.read_csv(combined_data_path)

        # make the protocol for adjusting log values
        log_adj = max((-1 * min(np.min(data["Price"]), np.min(data["Last Price"]))) + 1, log_adj)

        
        data["Date"] = pd.to_datetime(data["Date"])
        pre_covid = data.loc[data["Date"] < COVID_START].copy()

        inter_crisis = data.loc[data["Date"] > COVID_END].copy()
        inter_crisis = inter_crisis[inter_crisis["Date"] < WAR_START].copy()
        pre_covid = pd.concat([pre_covid, inter_crisis])

        covid = data.loc[data["Date"] > COVID_START].copy()
        covid = covid.loc[covid["Date"] < COVID_END].copy()

        war = data[data["Date"] > WAR_START].copy()

        # run the regressions on the given datasets
        
        output_str = ""
        
        for start in time_blocks.keys():
                output_str += "Coefficients for {} to {} \n".format(start, time_blocks[start])

                output_str += "Pre-COVID in {} \n".format(country_name)
                string_add, df_pre = aid_regression(pre_covid, country_name, log_adj, time_start=start, time_end = time_blocks[start], graphs = False, verbose = False) 
                output_str += string_add


                output_str += "COVID Era in {} \n".format(country_name)
                string_add, df_covid = aid_regression(covid, country_name, log_adj, time_start=start, time_end = time_blocks[start], graphs = False, verbose = False) 
                output_str += string_add

                output_str += "War Era in {} \n".format(country_name)
                string_add, df_war = aid_regression(war, country_name, log_adj, time_start=start, time_end = time_blocks[start], graphs = False, verbose = False) 
                output_str += string_add
                output_str += "\n"
        df_pre["Era"] = "red"
        df_covid["Era"] = "green"
        df_war["Era"] = "purple"
        df = pd.concat([df_pre, df_covid, df_war]) 
        df.set_index("Timeseries", inplace=True)
        df.sort_index(inplace=True)
        print(df)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=df["Actual"], name = "Actual"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Pred"], mode= "lines+markers", marker = dict(color = df["Era"].values), name = "Prediction"))
        

        fig.update_yaxes(title = "Marginal Price (€/MWh)")
        fig.update_xaxes(title = "Date")
        fig.update_layout(legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                        ), title = country_name, font = dict(size=20))
        
        COVID_END = pd.to_datetime(COVID_END)
        war_start= 500
        if country_name == "Spain": 
                war_start = 300

        fig.add_annotation(x = COVID_START, y= 250, text = "Covid Start", showarrow=False)
        fig.add_annotation(x = COVID_END, y=250, text = "Covid End", showarrow=False)
        fig.add_annotation(x = WAR_START, y= war_start, text = "War Start", showarrow=False)


        fig.show()
        output_str += "Overall R^2: {}".format(sklearn.metrics.r2_score(df["Actual"], df["Pred"]))

        print(output_str)
        return output_str

""" iteration_helper: helper function for betas_over_time to enable parallelization

Keyword Arguments:
start_date: date to start analysis
dataset: full dataset
timeperiod_length: length of the timeperiod to consider
roll_forward: timeperiod frequency to roll forward between periods
log_adj: the log adjustment

Returns a dictionary with the start date, LNG Beta, and demand beta
"""
def processing(data, country_name, timeperiod_length, roll_forward, log_adj, time_blocks, start_date):
                if(time_blocks):
                        time_blocks = {0:6, 7:11, 12:16, 17:20, 21:23}
                else:
                        time_blocks = {0:23}

                end = start_date+dt.timedelta(days=timeperiod_length)

                period = data[data["Date"] > start_date].copy()
                period = period[period["Date"] < end].copy()

                period_row = []
                #coefs_period = filter_and_regress(period, country_name, log_adj=log_adj, verbose=False)
                for key in time_blocks:
                        coefs_period = unfiltered_regression(period, country_name, log_adj, time_start=key, time_end=time_blocks[key])
                        X, actual = get_X(period, log_adj, time_start = key, time_end = time_blocks[key])
                        score = coefs_period.score(X, actual)
                        period_row.append({"Date": end, "Start": key, "LNG Beta": coefs_period.coef_[0], 
                                           "Coal Beta": coefs_period.coef_[1], "Demand Beta": coefs_period.coef_[2], "R^2": score})
                
                return period_row



""" betas_over_time: generate a plot of betas for the ng and demand terms using shortened rolling timeperiods

Keyword Arugments:
combined_data_path: path to the combined data frame in filesystem
country_name: country under question for labelling
log_adj: the minimal possible adjustment for all logs
timeperiod_length: the length of the timeperiod to consider in the lookback for the regression (days)
roll_forward: the timeperiod frequency to roll forward between periods

Returns a dataframe of the betas for the features over time
"""
def betas_over_time(country_name, combined_data_path = " ", log_adj=1, timeperiod_length=730, roll_forward=25, produce_graphs = "a"):
        combined_data_path = "/Users/Ethan/Dev/Thesis Work/Data/{}/combined_data.csv".format(country_name)
        # read in the data from the combined dataset
        data = pd.read_csv(combined_data_path)

        log_adj = max((-1 * min(np.min(data["Price"]), np.min(data["Last Price"]))) + 2, log_adj)
        data["Date"] = pd.to_datetime(data["Date"])
        
        first_day = data["Date"].min()
        last_day = data["Date"].max() - dt.timedelta(timeperiod_length)
        
        date_range = pd.date_range(start=first_day, end=last_day, freq="{}D".format(roll_forward))
        
        beta_list = []
        
        partial_processing= partial(processing, data, country_name, timeperiod_length, roll_forward, log_adj, True)
        """
        for start in date_range:
                end = start+dt.timedelta(days=timeperiod_length)

                period = data[data["Date"] > start].copy()
                period = period[period["Date"] < end].copy()

                coefs_period = filter_and_regress(period, country_name, log_adj=log_adj, verbose=False)

                period_row = {"Date": start, "Demand Beta": coefs_period.coef_[0], "LNG Beta": coefs_period.coef_[1]}
                beta_list.append(period_row)
        """
        with multiprocessing.Pool() as pool:
                beta_list = pool.map(partial_processing, date_range)

        flat_beta_list = [item for sublist in beta_list for item in sublist]
        betas_over_time = pd.DataFrame(flat_beta_list)
        print(betas_over_time)
        betas_over_time["Combined"] = betas_over_time["LNG Beta"] + betas_over_time["Coal Beta"]
        if(produce_graphs == "a"):
                fig = make_subplots(rows=2, cols=2, subplot_titles = ("LNG", "Demand", "Coal", "R^2"))
                fig.add_trace(go.Scatter(x = betas_over_time["Date"], y=betas_over_time["LNG Beta"]
                                         #, title = "{} LNG Betas Over Time".format(country_name)
                                         ), row=1, col=1)
                #fig.show()
                fig.add_trace(go.Scatter(x = betas_over_time["Date"], y=betas_over_time["Demand Beta"]
                                         #, title = "{} Demand Betas Over Time".format(country_name)
                                         ), row=1, col=2)
                #fig.show()
                fig.add_trace(go.Scatter(x = betas_over_time["Date"], y=betas_over_time["Coal Beta"]
                                         #, title = "{} Coal Betas Over Time".format(country_name)
                                ), row=2, col=1)
                #fig.show()
                fig.add_trace(go.Scatter(x = betas_over_time["Date"], y=betas_over_time["R^2"]
                                         #, title = "{} R^2 Values Over Time".format(country_name)
                                         ), row=2, col=2)
                fig.update_layout(title_text = "{} Betas over time".format(country_name)) 
                fig.show()
        elif(produce_graphs == "b"):
                grouped_data = data.copy()
                grouped_data = grouped_data.set_index("Date")
                grouped_data = grouped_data.resample("{}D".format(roll_forward)).mean()
                fig = make_subplots(rows=2, cols=1, shared_xaxes = True,  subplot_titles = ("LNG", "Coal"),
                                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x= betas_over_time["Date"], y=betas_over_time["LNG Beta"], name="LNG Beta"),
                              row=1, col=1, secondary_y = False)
                
                fig.add_trace(go.Scatter(x = grouped_data.index, y = grouped_data["Last Price"], name="LNG Price"),
                              row=1, col=1, secondary_y = True)
                
                fig.add_trace(go.Scatter(x = betas_over_time["Date"], y = betas_over_time["Coal Beta"], name = "Coal Beta"),
                              row=2, col=1, secondary_y = False)
                fig.add_trace(go.Scatter(x = grouped_data.index, y = grouped_data["Coal Price_x"], name="Coal Price"),
                              row=2, col=1, secondary_y = True)
                
                fig.update_yaxes(title_text = "LNG Beta", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text = "LNG Price", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text = "Coal Beta", row=2, col=1, secondary_y=False)
                fig.update_yaxes(title_text = "Coal Price", row=2, col=1, secondary_y=True)

                fig.update_layout(title="Prices vs. Betas over time in {}".format(country_name))
                fig.show()
        elif(produce_graphs == "c"):
                fig = make_subplots(rows=2, cols=2, subplot_titles = ("LNG", "Demand", "Coal", "R^2"))
                for start in betas_over_time["Start"].unique():
                        group_data = betas_over_time[betas_over_time["Start"] == start]
                        fig.add_trace(go.Scatter(x = group_data["Date"], y=group_data["LNG Beta"], name = "{}".format(start)
                                                #, title = "{} LNG Betas Over Time".format(country_name)
                                                ), row=1, col=1)
                        #fig.show()
                        fig.add_trace(go.Scatter(x = group_data["Date"], y=group_data["Demand Beta"], name = "{}".format(start)
                                                #, title = "{} Demand Betas Over Time".format(country_name)
                                                ), row=1, col=2)
                        #fig.show()
                        fig.add_trace(go.Scatter(x = group_data["Date"], y=group_data["Coal Beta"], name = "{}".format(start)
                                                #, title = "{} Coal Betas Over Time".format(country_name)
                                        ), row=2, col=1)
                        #fig.show()
                        fig.add_trace(go.Scatter(x = group_data["Date"], y=group_data["R^2"], name = "{}".format(start)
                                                #, title = "{} R^2 Values Over Time".format(country_name)
                                                ), row=2, col=2)
                fig.update_layout(title_text = "{} Betas over time".format(country_name)) 
                fig.show() 

        return betas_over_time


"""scatter_gen: generate a scatter plot for the given country between percent natural gas used and elec price

Keyword Arguments:
country_name: name of country to plot

returns null
"""
def scatter_gen(country_name):

        # first read in the generator information and combine it with the gas information
        gen_df = gen_read_clean(country_name)
        aggregated_df = combine_gen_gas(country_name, gen_df, freq="1H")

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Gas Total vs. Elec Price", 
                                                        "Gas Percent vs. Elec Price"))
        
        fig.add_trace(go.Scatter(x=aggregated_df["Total"], y=aggregated_df["Price"], mode="markers", opacity=0.4, 
                                text = aggregated_df["Start"], hovertemplate="%{text}"),
                        row=1, col=1)
        
        fig.add_trace(go.Scatter(x=aggregated_df["Percent Gas"], y=aggregated_df["Price"], mode="markers", opacity=0.4, 
                                text = aggregated_df["Start"], hovertemplate="%{text}"),
                        row=1, col=2)
        
        fig.update_layout(height=800, width=1300, title_text=country_name)

        fig.show()


def get_covid_df(country_name):
        covid_df = pd.read_csv("/Users/ethan/Dev/Thesis Work/Data/vaccinations.csv")

        country_covid_df = covid_df[covid_df["location"] == country_name]
        country_covid_df["date"] = pd.to_datetime(country_covid_df["date"])

        return country_covid_df

""" incorporate_covid_data: add the data from the covid dataset to the existing electricity price data

Keyword Arguments:
country_name: name of the country under question

returns a dataframe including various vaccine rate data about the country in question.
"""
def incorporate_covid_data(country_name):
        country_covid_df = get_covid_df(country_name)

        elec_df = pd.read_csv("/Users/Ethan/Dev/Thesis Work/Data/{}/combined_data.csv".format(country_name))
        elec_df["Date"] = pd.to_datetime(elec_df["Date"])
        #elec_df = elec_df.set_index("Date")

        #elec_df = elec_df.groupby(pd.Grouper(freq="1W")).mean()

        merged_df = pd.merge(elec_df, country_covid_df, right_on="date", left_on="Date", how="left", sort=True)
        
        merged_df = merged_df[merged_df["date"] > "2020-03-01"]
        merged_df["people_vaccinated"] = merged_df["people_vaccinated"].ffill()
        merged_df = merged_df.fillna(0)

        return merged_df


""" covid_graphs: make various graphs about the covid data that I presently find interesting 

Keyword Arguments:
country_name: name of the country to graph

displays the current graphs that I'm interested in for the given country
"""
def covid_graphs(country_name):
        #df = incorporate_covid_data(country_name)
        
        """
        fig = px.scatter(df, x = "people_vaccinated", y="Price", title=country_name)
        fig.show()
        """
        #df["Date"] = pd.to_datetime(df["Date"])
        #df = df.set_index("Date")
        #df = df.groupby(pd.Grouper(freq="1W")).mean()
        
        covid_df = get_covid_df(country_name)

        beta_df = betas_over_time(country_name, roll_forward=7, produce_graphs=False)

        merged_df = pd.merge_asof(beta_df, covid_df, left_on="Date", right_on="date", tolerance = dt.timedelta(weeks=2))

        merged_df = merged_df.fillna(0)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
                go.Scatter(x=merged_df["Date"], y=merged_df["Demand Beta"]), secondary_y=False
        )

        fig.add_trace(
                go.Scatter(x=merged_df["Date"], y=merged_df["people_vaccinated"]), secondary_y=True
        )

        fig.show()
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
                go.Scatter(x=df.index, y=df["Price"], name = "Electricity Price"),
                secondary_y=False
        )

        fig.add_trace(
                go.Scatter(x=df.index, y=df["people_vaccinated"], name="Vaccinations"),
                secondary_y=True
        )

        fig.show()
        """

""" score_model: fits a given linear model for a given country

Keyword Arguments:
country_name: country name for figure heading and accessing data
group_freq: the frequency to group by in stadard dt coding
log_adj: minimum value for log adjustment. Only use if errors persist.

prints scoring information and displays figure
returns null
"""
def score_model(country_name, group_freq="1W", log_adj=1):
        data = pd.read_csv(os.path.join("/Users","Ethan","Dev","Thesis Work","Data",country_name,"combined_data.csv"))

        log_adj = max((-1 * min(np.min(data["Price"]), np.min(data["Last Price"]))) + 2, log_adj)
        model = filter_and_regress(data, country_name, log_adj=log_adj)

        X = pd.DataFrame()
        X["Log LNG"] = np.log(data["Last Price"] + log_adj)
        X["Log Coal"] = np.log(data["Coal Price_x"] + log_adj)
        X["Forecast"] = data["Forecast"]
        
        
        actual = np.log(data["Price"] + log_adj)
        print(model.score(X, actual))


def get_X(dataframe, log_adj, time_start=0, time_end=23):
        dataframe["Start"] = pd.to_datetime(dataframe["Start"])

        dataframe = dataframe[dataframe["Start"].dt.hour >= time_start]
        dataframe = dataframe[dataframe["Start"].dt.hour <= time_end]

        X = pd.DataFrame()
        X["Log LNG"] = np.log(dataframe["Last Price"] + log_adj)
        X["Log Coal"] = np.log(dataframe["Coal Price_x"] + log_adj)
        X["Forecast"] = dataframe["Forecast"]

        # properly break into the daily elements based on the period of the day considered.

        actual = np.log(dataframe["Price"] + log_adj)
        
        return X, actual


""" intraday_price_plot: make a plot of the intraday price averages in the country

country_name: name of country

returns dataframe of intraday price averages
"""
def intraday_price_plot(country_name):
        data = pd.read_csv(os.path.join("/Users","Ethan","Dev","Thesis Work","Data",country_name,"combined_data.csv"))
        
        data["Start"] = pd.to_datetime(data["Start"])

        data.set_index("Start", inplace=True)

        data_intraday_mean = data.groupby([data.index.time]).mean(numeric_only=True)

        fig = make_subplots(rows = 2, cols=1, subplot_titles = ["Demand Forecast", "Pricing"])

        fig.add_trace(go.Scatter(x=data_intraday_mean.index, y = data_intraday_mean["Forecast"], 
                     name = "Demand Forecast"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_intraday_mean.index, y = data_intraday_mean["Price"], 
                     name = "Pricing"), row=2, col=1)

        fig.update_layout(title_text = country_name)

        fig.show()


        return data