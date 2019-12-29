# -*- coding: utf-8 -*-
"""Draw_Plotly.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QvTcxAB0_5nPkBk0-bnh6lCyJMWwRVS_
"""

def load_img2graphdict(filename):
        i2gdict = {}
        with open(filename, mode="r") as file:
            for line in file:
                idx, rest = line.strip().strip(";").split(":")
                idx = int(idx)
                x, y = rest.split(",")
                x = int(x)
                y = int(y)
                i2gdict[idx] = (x, y)
        return i2gdict

img2graphdict = load_img2graphdict("graf2img.txt")

import plotly.graph_objects as go


def draw_map(filename, fig):
  img2graphdict = load_img2graphdict(filename)

  fig.add_trace(go.Scatter(
    x=[a[0] for a in img2graphdict.values()],
    y=[a[1] for a in img2graphdict.values()],
    mode="markers",
    hovertext=[str(a) for a in img2graphdict.keys()]
  ))

  fig.add_scatter(
    x=[],
    y=[],
    mode="lines+markers"
  )
  fig.show()


def draw_trace(path, fig):
  x1=[]
  y1=[]
  for i in path:
    a,b=(img2graphdict[i])
    x1.append(a)
    y1.append(b)
  fig.update_traces(
    x=x1,y=y1,
    selector=dict(type="scatter", mode="lines+markers"))
  fig.show()

from PIL import Image

image=Image.open("def.jpg")
layout= go.Layout(images= [dict(
                  source= image,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")])

filename="graf2img.txt"
fig = go.Figure(layout=layout)
draw_map(filename,fig)

path=[1,2,3,13,14,15,16,26]
draw_trace(path,fig)


