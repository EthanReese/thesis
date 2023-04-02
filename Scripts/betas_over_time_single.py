from ElecEuro import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go

countries = ["Austria", "Spain", "France", "Netherlands", "Germany", "Italy"]




if __name__=="__main__":
        demand = pd.DataFrame()
        lng_beta = pd.DataFrame()
        coal_beta = pd.DataFrame()
        R2 = pd.DataFrame()
        for country in countries:
                df = betas_over_time(country, log_adj=1, timeperiod_length=60, roll_forward=10, produce_graphs=False)
                demand["Date"] = df["Date"]
                lng_beta["Date"] = df["Date"]
                coal_beta["Date"] = df["Date"]
                R2["Date"] = df["Date"]
                demand[country] = df["Demand Beta"]
                lng_beta[country] = df["LNG Beta"]
                coal_beta[country] = df["Coal Beta"]
                R2[country] = df["R^2"]
        
        demand_fig = go.Figure()
        lng_fig = go.Figure()
        coal_fig = go.Figure()
        R2_fig = go.Figure()
        for country in countries:
                demand_fig.add_trace(go.Scatter(x=demand["Date"], y=demand[country], name=country))
                lng_fig.add_trace(go.Scatter(x=lng_beta["Date"], y=lng_beta[country], name=country))
                coal_fig.add_trace(go.Scatter(x=coal_beta["Date"], y=coal_beta[country], name=country))
                R2_fig.add_trace(go.Scatter(x=R2["Date"], y=R2[country], name=country))
        demand_fig.update_layout(title="Demand Betas Over Time")
        lng_fig.update_layout(title="LNG Betas Over Time")
        coal_fig.update_layout(title="Coal Betas Over Time")
        R2_fig.update_layout(title="R^2 Over Time")
        demand_fig.show()
        lng_fig.show()
        coal_fig.show()
        R2_fig.show()
