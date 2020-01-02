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
from dash.dependencies import Input, Output








path = []
img2graphdict = load("assets\\graf2img.txt")


image = Image.open("assets/def.jpg")
fig = px.imshow(image)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)



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
    ], className='image',

    style={'width': '30%', 'display': 'inline-block', 'float': 'right', 'vertical-align': 'top'}),
    dcc.Dropdown(
        id='dropdown_user',
        options=[
            {'label': 'User1', 'value': 'User1'},
            {'label': 'User2', 'value': 'User2'},
            {'label': 'User3', 'value': 'User3'},
            {'label': 'User4', 'value': 'User4'},
            {'label': 'User5', 'value': 'User5'}
        ], className='dropdown_user',
    style={'width': '18%', 'display': 'inline-block', 'vertical-align': 'top', 'float': 'left'}),
dcc.Dropdown(
        id='dropdown_routes',
        options=[
            {'label': 'Route1', 'value': 'Route1'},
            {'label': 'Route2', 'value': 'Route2'},
            {'label': 'Route3', 'value': 'Route3'},
            {'label': 'Route4', 'value': 'Route4'},
            {'label': 'Route5', 'value': 'Route5'}
        ], className='dropdown_route',
    style={'width': '18%', 'display': 'inline-block', 'vertical-align': 'top'}),
dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Order1', 'value': 'Order1'},
            {'label': 'Order2', 'value': 'Order2'},
            {'label': 'Order3', 'value': 'Order3'},
            {'label': 'Order4', 'value': 'Order4'},
            {'label': 'Order5', 'value': 'Order5'}
        ], className='dropdown',
    style={'width': '18%', 'display': 'inline-block', 'vertical-align': 'top'}),
    html.Div(id='output')
],style={'backgroundColor':'blue'})


@app.callback(
    Output('map', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_figure(value):
    path=[]
    if value=='Order1':
        path = [1,2,3,4,14,24,25,26]
    if value=='Order2':
        path=[35,45,55,56]
    draw_trace(path,fig,img2graphdict)
    return fig




if __name__ == "__main__":
    app.run_server(debug=True)


