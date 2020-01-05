import dash
import dash_core_components as dcc
import dash_html_components as html
from load_img2graphdict import load_img2graphdict as load
from Draw_trace import draw_trace
import plotly.express as px
from PIL import Image
from dash.dependencies import Input, Output
import flask
import dash_table
import json


with open('assets/final_dict.json', 'r') as f:
    display_dict = json.load(f)

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

data=[]
draw_trace(path,fig,img2graphdict)


_app_route = '/dash-core-components/logoutbutton'


@app.server.route('/custom-auth/login', methods=['POST'])
# Create a login route
def route_login():
    data = flask.request.form
    username = data.get('username')
    password = data.get('password')

    rep = flask.redirect(_app_route)
    rep.set_cookie('custom-auth-session', username)
    return rep

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
            options=[],
            className='dropdown1'),
        dcc.Dropdown(
            id='dropdown_order',
            options=[
                {'label': 'Order1', 'value': 0},
                {'label': 'Order2', 'value': 1},
                {'label': 'Order3', 'value': 2},
                {'label': 'Order4', 'value': 3},
                {'label': 'Order5', 'value': 4},
                {'label': 'Order6', 'value': 5}
            ],
            value=1,
            className='dropdown2'),
    ], className="Dropdowns"),

    html.Div([
        html.Div([
            dash_table.DataTable(
                id='table',
                columns=[
                    {'id': 'Item', 'name': 'Item'},
                    {'id': "Location", 'name': "Location"},
                    {'id': "Delivery", 'name': "Delivery"},
                    {'id': "Order", 'name': "Order"},
                    {'id': "Nod_number", 'name': "Nod_number"},
                         ],
                data=data,
                )
            ], className='table'),

        html.Div([
            dcc.Graph(
                id='map',
                figure=fig
            )
        ], className='image'),
    ], className='Data')
])

@app.callback(Output('dropdown_user', 'options'),
              [Input('custom-auth-frame', 'id')])
def update_dropdown(_):
    session_cookie = flask.request.cookies.get('custom-auth-session')

    if not session_cookie:
        options = []
        return options
    else:
        if session_cookie == 'User1':
            options = [
                {'label': 'User1', 'value': '1'}]
        if session_cookie == 'User2':
            options = [
                {'label': 'User2', 'value': '2'}]
        if session_cookie == 'User3':
            options = [
                {'label': 'User3', 'value': '3'}]
        if session_cookie == 'User4':
            options = [
                {'label': 'User4', 'value': '4'}]
        if session_cookie == 'User5':
            options = [
                {'label': 'User5', 'value': '5'}]
        if session_cookie == 'Admin':
            options = [
                {'label': 'User1', 'value': '1'},
                {'label': 'User2', 'value': '2'},
                {'label': 'User3', 'value': '3'},
                {'label': 'User4', 'value': '4'},
                {'label': 'User5', 'value': '5'}
                ]
        return options

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
    [dash.dependencies.Input('dropdown_user', 'value'),
        dash.dependencies.Input('dropdown_order', 'value')
     ])
def update_figure(value1, value2):
    if not value1:
        path = []
        draw_trace(path, fig, img2graphdict)
        return fig
    else:
        path = display_dict[str(value1)][value2]['optimal_route']
        draw_trace(path, fig, img2graphdict)
        return fig


@app.callback(
    Output('table', 'data'),
    [dash.dependencies.Input('dropdown_user', 'value'),
        dash.dependencies.Input('dropdown_order', 'value')
     ])
def update_table(value1, value2):
    if not value1:
        data = []
        return data
    dictionary = display_dict[str(value1)][value2]["node_dict"]
    for key_node, node in dictionary.items():
        if not isinstance(node, list):
            dictionary[key_node] = [node]
    data = [
        {"Item": item["item"]["product_name"],
            "Location": item["location"],
            "Order": item["order_id"],
            "Nod_number": key_node,
            "Delivery": item["delivery_option"]
         } for key_node, node in dictionary.items() for single in node for id, item in single.items()]
    data1 = []
    for i in range(len(display_dict[value1][value2]['optimal_nodes_list'])):
        for nods in data:
            if str(nods['Nod_number']) == str(display_dict[value1][value2]['optimal_nodes_list'][i]):
                data1.append(nods)
    return data1


if __name__ == '__main__':
    app.run_server(debug=True)