import dash
from dash import dcc, html, Input, Output, State, ctx
import dash.exceptions
import numpy as np
import plotly.graph_objs as go
import sympy
from dash_bootstrap_templates import load_figure_template

load_figure_template("darkly")

x = sympy.symbols('x')

app = dash.Dash(__name__)
server = app.server 

app.layout = html.Div(
    children=[
        html.Div(
            className='row',
            children=[
                #seccion de configuracion de la grafica
                html.Div(
                    className='four columns div-user-controls',
                    children=[
                        html.H2("Visualización de Convergencia de Series de Fourier", style={'fontWeight': 'bold', 'color': 'white'}),

                        html.Label("Función f(x) (usa sintaxis de SymPy, e.g. x**2 o Piecewise):"),

                        #caja para escribir la funcion
                        dcc.Textarea(
                            id='func-input',
                            value='Piecewise((x, (x >= 0) & (x <= 1)), (-1, (x > 1) & (x <= 2)))',
                            style={"backgroundColor": "#383434",
                                "color": "white",
                                "border": "1px solid #555",
                                "borderRadius": "5px",
                                "padding": "10px",
                                "width": "100%",
                                "height": "100px",
                                "fontSize": "15px",
                                'marginTop': '10px',
                                'marginBottom': '20px'}
                        ),
                        
                        #cajas para introducir el intervalo de la funcion
                        html.Div([
                            html.Label('Intervalo de la función', style={"fontWeight": "bold"}),
                            html.Div([
                                html.Label('a =', style={"fontWeight": "bold", "paddingTop": "10px"}),
                                dcc.Input(
                                    id='a-input',
                                    type='number',
                                    value=0,
                                    style={"backgroundColor": "#383434",
                                "color": "white",
                                "border": "1px solid #555",
                                "borderRadius": "5px",
                                "padding": "10px",
                                "width": "80px"}
                                ),
                                html.Label('b =', style={"fontWeight": "bold", "paddingTop": "5px"}),
                                dcc.Input(
                                    id='b-input',
                                    type='number',
                                    value=2,
                                    style={"backgroundColor": "#383434",
                                "color": "white",
                                "border": "1px solid #555",
                                "borderRadius": "5px",
                                "padding": "10px",
                                "width": "80px"}
                                )
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '20px'}),
                        ]),
                        
                        #boton para actualizar la funcion
                        html.Button("Actualizar función", id="update-button"),
                        html.Hr(),
                        
                        #lista de los valores de n para calcular las series de Fourier
                        html.Label("Lista de valores de n", style={"fontWeight": "bold"}),
                        html.Div([
                            html.Label("n =", style={"fontWeight": "bold", "paddingTop": "0px"}),
                            dcc.Input(
                                id='n-input',
                                type='number',
                                placeholder='1',
                                min=1,
                                max=100,
                                style={"backgroundColor": "#383434",
                                "color": "white",
                                "border": "1px solid #555",
                                "borderRadius": "5px",
                                "padding": "10px",
                                "width": "80px"}
                            ),
                            html.Button('Agregar', id='add-n', n_clicks=0),
                            html.Button('Limpiar', id='clear-n', n_clicks=0),
                        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
                        
                        html.Div(id='n-list-display', style={'marginTop': '10px'}),
                        dcc.Store(id='n-store', data=[1, 5, 10]),
                        html.Hr(),
                        
                        #toggle para mostrar la banda de tolerancia
                        html.Label("Mostrar bandas de tolerancia", style={"fontWeight": "bold"}),
                        dcc.Checklist(['Mostrar bandas'], id='tolerance-check', value=[])
                    ]
                ),

                #seccion de la grafica
                html.Div(
                    className='eight columns div-for-charts bg-grey',
                    children=[dcc.Graph(id='fourier-graph', figure=go.Figure().update_layout(
        # Hide all visual elements
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False  # Hides axis line, ticks, and numbers
        },
        yaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False
        },
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent
        paper_bgcolor='rgba(0,0,0,0)',
        margin={'t': 0, 'b': 0, 'l': 0, 'r': 0}  # Remove padding
    ))]
                )
            ]
        )
    ]
)

dcc.Loading(
        id='loading-graph',
        type='circle',  # Shows spinner while loading
        children=[dcc.Graph(id='fourier-graph')]
    )

@app.callback(
    Output('n-store', 'data'),
    Input('add-n', 'n_clicks'),
    Input('clear-n', 'n_clicks'),
    State('n-input', 'value'),
    State('n-store', 'data'),
    prevent_initial_call=True
)
#funcion para actualizar la lista de los valores de n
def update_n_list(add_clicks, clear_clicks, new_n, n_list):
    if ctx.triggered_id == 'clear-n':
        return []
    elif ctx.triggered_id == 'add-n' and new_n is not None:
        if new_n not in n_list:
            return sorted(n_list + [new_n])
    return n_list

@app.callback(
    Output('n-list-display', 'children'),
    Input('n-store', 'data')
)
def display_n_list(n_list):
    return html.Ul([html.Li(f"n = {n}") for n in n_list]) if n_list else "Ningún valor agregado."

#calcula los coeficientes de la serie de Fourier
def compute_coeffs(f_expr, N, L, a, b):
    a0 = (1/L) * sympy.integrate(f_expr, (x, a, b))
    an = [(1/L) * sympy.integrate(f_expr * sympy.cos(n * sympy.pi * x / L), (x, a, b)) for n in range(1, N+1)]
    bn = [(1/L) * sympy.integrate(f_expr * sympy.sin(n * sympy.pi * x / L), (x, a, b)) for n in range(1, N+1)]
    return float(a0), [float(a) for a in an], [float(b) for b in bn]

#calcula la suma de Fourier
def fourier_sum(x_vals, a0, an, bn, N, L):
    result = a0 / 2 * np.ones_like(x_vals)
    for n in range(1, N + 1):
        result += an[n-1] * np.cos(n * np.pi * x_vals / L) + bn[n-1] * np.sin(n * np.pi * x_vals / L)
    return result

@app.callback(
    Output('fourier-graph', 'figure'),
    Input('update-button', 'n_clicks'),
    Input('n-store', 'data'),
    Input('tolerance-check', 'value'),
    State('func-input', 'value'),
    State('a-input', 'value'),
    State('b-input', 'value'),
    prevent_initial_call=False
)
#se grafica la funcion introducida por el usuario, asi como las aproximaciones de Fourier de la misma funcion
def update_graph(n_clicks, n_values, show_band, func_text, a, b, epsilon = 0.1):
    try:
        f_expr = sympy.sympify(func_text)
        f_func = sympy.lambdify(x, f_expr, modules=["numpy"])
    except Exception as e:
        return go.Figure().update_layout(title="Error en la función ingresada", annotations=[
            dict(text=str(e), showarrow=False, font=dict(color="red"))
        ], template='plotly_black')
    a = float(a)                     #principio del intervalo
    b = float(b)                     #final del intervao
    L = (b-a)/2                      #longitud de la mitad del intervalo
    x_vals = np.linspace(a, b, 300)  #300 valores de x para graficar la funcion
    y_vals = f_func(x_vals)          #funcion evaluada en los 300 valores de x
    try:
        x_plot = x_vals.tolist()
        y_plot = y_vals.tolist()
    except Exception:
        y_plot = np.full_like(x_vals, np.nan)

    fig = go.Figure() 
    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)", line=dict(color="white"))) #se grafica la funcion original
    #fig.layout.template = 'plotly_dark'


    #se grafica la aproximacion de Fourier para cada n en la lista
    for N in sorted(n_values):
        try:
            a0, an, bn = compute_coeffs(f_expr, N, L, a, b)
            y_approx = fourier_sum(x_vals, a0, an, bn, N, L)
            fig.add_trace(go.Scatter(x=x_vals, y=y_approx, name=f'S_{N}(x)'))
        except Exception:
            continue
    
    #seccion para mostrar la banda de epsilon
    if 'Mostrar bandas' in show_band:
        y_above = []
        y_below = []
        for y in y_plot:
            try:
                y_above.append(y + epsilon)
                y_below.append(y - epsilon)
            except: 
                y_above.append(y)
                y_below.append(y)
        fig.add_trace(go.Scatter(x=x_vals, y=y_above, name="f(x)+ε", line=dict(dash="dash", color="green")))
        fig.add_trace(go.Scatter(x=x_vals, y=y_below, name="f(x)-ε", line=dict(dash="dash", color="green")))

    fig.update_layout(xaxis_title=r"x", yaxis_title="y", hovermode='x unified')
    return fig

if __name__ == '__main__':
    app.run(debug=True)
