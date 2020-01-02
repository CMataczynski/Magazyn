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
import flask

app = dash.Dash(__name__)
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


_app_route = '/dash-core-components/logoutbutton'

@app.server.route('/custom-auth/login', methods=['POST'])
# Create a login route
def route_login():
    data = flask.request.form
    username = data.get('username')
    password = data.get('password')

    if username=="User1" and password=="User1":
        User=username

    rep = flask.redirect(_app_route)
    rep.set_cookie('custom-auth-session', username)
    return rep

# create a logout route
@app.server.route('/custom-auth/logout', methods=['POST'])
def route_logout():
    # Redirect back to the index and remove the session cookie.
    rep = flask.redirect(_app_route)
    rep.set_cookie('custom-auth-session', '', expires=0)
    return rep


# Simple dash component login form.
login_form = html.Div([
    html.Form([
        dcc.Input(placeholder='username', name='username', type='text'),
        dcc.Input(placeholder='password', name='password', type='password'),
        html.Button('Login', type='submit')
    ], action='/custom-auth/login', method='post')
])


app.layout = html.Div([
    html.Div([
        html.H2("System optymalizacji zamówień w magazynie"),
        html.H3(id='custom-auth-frame')
    ], className="banner"),

    html.Div([
        dcc.Dropdown(
            id='dropdown_user',
            options=[
                {'label': 'User1', 'value': 'User1'},
                {'label': 'User2', 'value': 'User2'},
                {'label': 'User3', 'value': 'User3'},
                {'label': 'User4', 'value': 'User4'},
                {'label': 'User5', 'value': 'User5'}
            ], className='dropdown1'),
        dcc.Dropdown(
            id='dropdown_order',
            options=[
                {'label': 'Order1', 'value': 'Order1'},
                {'label': 'Order2', 'value': 'Order2'},
                {'label': 'Order3', 'value': 'Order3'},
                {'label': 'Order4', 'value': 'Order4'},
                {'label': 'Order5', 'value': 'Order5'}
            ], className='dropdown2'),
        html.Div(id='output'),
    ], className="Dropdowns"),


    html.Div([
        dcc.Graph(
            id='map',
            figure=fig
        )
    ], className='image')

])

@app.callback(Output('custom-auth-frame', 'children'),
              [Input('custom-auth-frame', 'id')])
def dynamic_layout(_):
    session_cookie = flask.request.cookies.get('custom-auth-session')

    if not session_cookie:
        return login_form

    return html.Div([
        dcc.LogoutButton(logout_url='/custom-auth/logout',)
    ], className='logout_button')


@app.callback(
    Output('map', 'figure'),
    [dash.dependencies.Input('dropdown_order', 'value')])
def update_figure(value):
    path=[]
    if value=='Order1':
        path = [1,2,3,4,14,24,25,26]
    if value=='Order2':
        path=[35,45,55,56]
    draw_trace(path,fig,img2graphdict)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
