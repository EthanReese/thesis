from ElecEuro import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go

countries = ["Austria", "Spain", "France", "Netherlands", "Germany", "Italy"]




if __name__=="__main__":
        
        for country in countries:
                df = betas_over_time(country, log_adj=1, timeperiod_length=365, roll_forward=30, produce_graphs="c")
