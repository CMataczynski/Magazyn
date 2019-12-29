import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_auth
from load_img2graphdict import load_img2graphdict as load
from Draw_trace import draw_trace
import plotly.express as px
from PIL import Image
import numpy as np

img2graphdict = load("assets\\graf2img.txt")
path = [1,2,3,4,14,24,25,26,27,37,47,48,49,59]


image = Image.open("assets/def.jpg")
fig = px.imshow(image)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)


# wyświetlanie wszystkich punktów
# fig.add_trace(go.Scatter(
#     x=[a[0] for a in img2graphdict.values()],
#     y=[a[1] for a in img2graphdict.values()],
#     mode="markers",
#     hovertext=[str(a) for a in img2graphdict.keys()]
#   ))

fig.add_scatter(
    x=[],
    y=[],
    mode="lines+markers"
  )

draw_trace(path,fig,img2graphdict)

# Autoryzacja
VALID_USERNAME_PASSWORD_PAIRS = {
    'Admin': "Admin",
    'User1': 'User1',
    'User2': 'User2'
}

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
app.layout = html.Div([
    html.Div([
        html.H2("System optymalizacji zamówień w magazynie"),
    ], className="banner"),
    html.Div([
        dcc.Graph(
            id='map',
            figure=fig
        )
    ], className='image'),
])

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

if __name__ == "__main__":
    app.run_server(debug=True)


