from ElecEuro import *
import plotly.express as px

if(__name__ == "__main__"):
        data = pd.read_csv("~/Dev/Thesis Work/Data/France/combined_data.csv")
        data["Start"] = pd.to_datetime(data["Start"])
        data.set_index("Start", inplace=True)
        data.sort_index(inplace=True)

        data = data.resample("3M").mean()
        fig = px.line(data, x=data.index, y="Total Ren")
        fig.show()