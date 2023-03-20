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
def read_and_clean(country, lng_path, area_suffix, coal_path = "./Bloomberg/Rotterdam Coal.xlsx",
                data_path = "/Users/Ethan/Dev/Thesis Work/Data"):
        init_directory = os.getcwd()
        # start by iterating and reading in all of the files
        price_data = []

        os.chdir(os.path.join(data_path, country, "Price"))
        for file in os.listdir():
                price_data.append(pd.read_csv(file))

        prices = pd.concat(price_data)

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
        os.chdir(os.path.join("..",".."))
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

        ## MERGING THE TWO DATASETS
        # merge the two sets of data
        elec_merged = pd.merge(left = prices, right = loads, on = "Start", how="left")
        elec_merged = elec_merged[["Start", "Price", "Actual", "Forecast"]]
        elec_merged["Date"] = pd.to_datetime(pd.to_datetime(elec_merged["Start"]).dt.date)
        ## READ IN LNG DATA FROM BLOOMBERG
        lng_data = pd.read_excel(lng_path)
        # filter columns
        lng_data = lng_data[["Date", "Last Price"]]

        ## READ IN COAL DATA FROM BLOOMBERG
        coal_data = pd.read_excel(coal_path)
        coal_data["Coal Price"] = coal_data["Last Price"]
        coal_data = coal_data[["Date", "Coal Price"]]
        # add the additional day to match up with the day ahead data
        combined = elec_merged.merge(lng_data, on="Date")
        combined = combined.merge(coal_data, on = "Date")

        combined = combined.dropna(subset=["Forecast", "Actual"])
        
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
        low_pass_percent=0.2, med_pass_percent=0.5, high_pass_percent=0.5, log_adj=20,
        verbose = True):
        
        combined_data = combined_data[["Last Price", "Price", "Forecast"]]
        combined_data = combined_data.dropna(axis=0)
        elec_price = combined_data["Price"].to_numpy()
        lng_price = combined_data["Last Price"].to_numpy()
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
        
        low_pass_elec = imf[:, low_pass_thresh_elec:]
        low_pass_means_elec = np.apply_along_axis(np.mean, 1, low_pass_elec)

        low_pass_lng = lng_imf[:, low_pass_thresh_lng:]
        low_pass_means_lng = np.apply_along_axis(np.mean, 1, low_pass_lng)

        low_pass_demand = demand_imf[:, low_pass_thresh_demand:]
        low_pass_means_demand = np.apply_along_axis(np.mean, 1, low_pass_demand)

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
        X_low = pd.DataFrame({"Log LNG": low_pass_means_lng, "Forecast": low_pass_means_demand})
        X_low_log = X_low.copy()
        X_low_log["Log LNG"] = X_low_log["Log LNG"].apply(lambda x: np.log(x+log_adj))
        low_model = linear_model.LinearRegression().fit(X_low_log, np.log(low_pass_means_elec+log_adj))
        
        """
        X_med = pd.DataFrame({"Log LNG": med_pass_means_lng, "Forecast": med_pass_means_demand})
        X_med_log = X_med.copy()
        X_med_log["Log LNG"] = X_med_log["Log LNG"].apply(lambda x: np.log(x+log_adj))
        med_model = linear_model.LinearRegression().fit(X_med_log, np.log(med_pass_means_elec + log_adj))
        """
        if(verbose):
                print("Low thresh LNG coefficient = {}, Demand Coefficient = {}".format(low_model.coef_[0], low_model.coef_[1]))
                print("Med thresh LNG coefficient = {}, Demand Coefficient = {}".format(med_model.coef_[0], med_model.coef_[1]))
        """
        X_high = pd.DataFrame({"LNG": high_pass_means_lng, "Demand": high_pass_means_demand})
        X_high_log = X_high.copy()
        X_high_log["LNG"] = X_high_log["LNG"].apply(lambda x: np.log(x + log_adj))
        high_model = linear_model.LinearRegression().fit(X_high_log, np.log(high_pass_means_elec + log_adj))
        print("High thresh LNG coefficient = {},  Demand Coefficent = {}".format(high_model.coef_[0], high_model.coef_[1]))
        """
        return low_model

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
# run regressions based on the timescales of COVID and war in Ukraine
def timeperiod_differences(combined_data_path, country_name, log_adj=1, COVID_START = "2020-03-01",
        COVID_END = "2021-08-01", WAR_START = "2022-02-01"):
        # these serve as best guesses, change at will
        # 50% of Europe was vaccinated by this date: 
        # https://www.bbc.com/news/explainers-52380823
        

        # read in the data from the combined dataset
        data = pd.read_csv(combined_data_path)

        # make the protocol for adjusting log values
        log_adj = max((-1 * min(np.min(data["Price"]), np.min(data["Last Price"]))) + 1, 30)

        
        data["Date"] = pd.to_datetime(data["Date"])
        pre_covid = data.loc[data["Date"] < COVID_START].copy()


        covid = data.loc[data["Date"] > COVID_START].copy()
        covid = covid.loc[covid["Date"] < COVID_END].copy()

        war = data[data["Date"] > WAR_START].copy()

        # run the regressions on the given datasets
        

        print("Pre-COVID in {}".format(country_name))
        coefs_pre = filter_and_regress(pre_covid, country_name, log_adj=log_adj)


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

""" iteration_helper: helper function for betas_over_time to enable parallelization

Keyword Arguments:
start_date: date to start analysis
dataset: full dataset
timeperiod_length: length of the timeperiod to consider
roll_forward: timeperiod frequency to roll forward between periods
log_adj: the log adjustment

Returns a dictionary with the start date, LNG Beta, and demand beta
"""
def processing(data, country_name, timeperiod_length, roll_forward, log_adj, start_date):
                end = start_date+dt.timedelta(days=timeperiod_length)

                period = data[data["Date"] > start_date].copy()
                period = period[period["Date"] < end].copy()

                coefs_period = filter_and_regress(period, country_name, log_adj=log_adj, verbose=False)

                period_row = {"Date": start_date, "Demand Beta": coefs_period.coef_[0], "LNG Beta": coefs_period.coef_[1]}
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
def betas_over_time(country_name, combined_data_path = " ", log_adj=1, timeperiod_length=720, roll_forward=25):
        combined_data_path = "/Users/Ethan/Dev/Thesis Work/Data/{}/combined_data.csv".format(country_name)
        # read in the data from the combined dataset
        data = pd.read_csv(combined_data_path)

        log_adj = max((-1 * min(np.min(data["Price"]), np.min(data["Last Price"]))) + 1, log_adj)
        data["Date"] = pd.to_datetime(data["Date"])
        
        first_day = data["Date"].min()
        last_day = data["Date"].max()
        
        date_range = pd.date_range(start=first_day, end=last_day, freq="{}D".format(roll_forward))
        
        beta_list = []
        
        partial_processing= partial(processing, data, country_name, timeperiod_length, roll_forward, log_adj)
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
        betas_over_time = pd.DataFrame(beta_list)
        fig = px.line(betas_over_time, x = betas_over_time["Date"], y=betas_over_time["LNG Beta"], title = 
                        "{} LNG Betas Over Time".format(country_name))
        fig.show()
        fig = px.line(betas_over_time, x = betas_over_time["Date"], y=betas_over_time["Demand Beta"], title = 
                        "{} Demand Betas Over Time".format(country_name))
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
        
        fig.add_trace(go.Scatter(x=aggregated_df["Total"], y=np.log(aggregated_df["Price"]), mode="markers", opacity=0.4, 
                                text = aggregated_df["Start"], hovertemplate="%{text}"),
                        row=1, col=1)
        
        fig.add_trace(go.Scatter(x=aggregated_df["Percent Gas"], y=np.log(aggregated_df["Price"]), mode="markers", opacity=0.4, 
                                text = aggregated_df["Start"], hovertemplate="%{text}"),
                        row=1, col=2)
        
        fig.update_layout(height=800, width=1300, title_text=country_name)

        fig.show()