
# Rysowanie ścieżki

def draw_trace(path, fig, img2graphdict):
  x1 = []
  y1 = []
  for i in path:
    a, b = (img2graphdict[i])
    x1.append(a)
    y1.append(b)
  fig.update_traces(
    x=x1, y=y1,
    hovertext=path,
    hoverinfo="text",
    selector=dict(type="scatter", mode="lines+markers"))
