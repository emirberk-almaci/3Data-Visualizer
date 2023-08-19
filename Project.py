"""Emirberk Almacı 150220751 Project Source Code"""
import io
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_daq as daq
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)
app.title = '3Data Visualizer'

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="3Data-Visualizer", className="header-title"
                        ),
                html.P(
                    children=(
                        "Visualize your CSV data"
                    ),
                    className="header-description",
                ),
            ],
            className="header"),

        html.Div(children=[
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A(' Select Files')],
                            className='file-load'),
                                ),
                    dcc.Dropdown(
                        options=['USD-TRY', 'Car Market', 'Laptop Price', 'World Population', 'Air Index'],
                        id='database-data',
                        placeholder='or Select Loaded Data',
                        className='database',),
                    html.Button('Clear uploaded data', id='clear-button', n_clicks=0,
                                className='clear_button'),
        ]),

        html.Div(
            [
                html.H3('Columns:', style={'textAlign': 'center'}),
                html.Ul(id='column-list', style={'textAlign': 'center'}),
                html.H3('Row Length:', style={'textAlign': 'center'}),
                html.Div(id='row-count-display', style={'textAlign': 'center'})
            ]
        ),

        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div('Select first attribute', style={'textAlign': 'center', 'color': '#18A2CB'}),
                        dcc.Dropdown(id='header-dropdown', value=''),
                        ]
                    ),
                html.Div(
                    children=[
                        html.Div('Select second attribute', style={'textAlign': 'center', 'color': '#18A2CB'}),
                        dcc.Dropdown(id='header-dropdown-2', value=''),
                             ]
                        ),

                html.Div(
                    children=[
                        html.Div('Select third attribute(for 3D)', style={'textAlign': 'center', 'color': '#18A2CB'}),
                        dcc.Dropdown(id='header-dropdown-3', value=''),
                             ]
                        ),

                    ],
            className='menu'),


        html.Div(
            children=[
                daq.ColorPicker(
                    id='color-picker',
                    label='Select Color',
                    value=dict(hex='#119DFF'),
                    className='color-picker'
                                ),

                html.Div(
                    children=[
                        html.Div('Select row range', style={'textAlign': 'center', 'color': '#18A2CB'}),
                        dcc.RangeSlider(
                                id='row-count', min=0, max=100, step=1, value=[0, 100],
                                tooltip={"placement": "bottom", "always_visible": True},
                                className='row-range'
                                ),
                        html.Div(id='slider-output-container', className='slider_output_container'),
                        html.Div('or Choose Random', className='random_sentence'),
                        dcc.Input(placeholder='Random data number', id='random-number', type='number', min=0,
                                  className='random_number')


                            ]
                ),


                html.Div(
                    children=[
                        html.Div('Select plot type', style={'textAlign': 'center', 'color': '#18A2CB'}),
                        dcc.Dropdown(options=['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'], id='select_plot',
                                     className='plot_type', value='', clearable=False),
                        html.Button('View summary statistics', id='statistic_button', n_clicks=0,
                                    className='statistic_button'),
                        html.Button('Submit', id='submit_button', n_clicks=0, className='submit_button'),
                        html.Div(id="alert-output", className='login_alert')
                             ]
                        ),

                    ],
            className='menu-2',
                ),

        html.Div(
            children=[
                html.Div(id='output-data-upload',
                            className='describe_table')]),


        html.Div(
            style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'textAlign': 'center'},
            children=[
                html.Div(
                    style={'width': '50%'},
                    children=[
                        html.Label('Graph Width:'),
                        dcc.Slider(
                            id='graph-width-slider',
                            min=300,
                            max=1000,
                            step=25,
                            value=800,
                            marks={300: '300', 400: '400', 500: '500', 600: '600', 700: '700',
                                   800: '800', 900: '900', 1000: '1000'},
                            className='graph_width_slider'
                                    )
                             ]
                        )
                    ]
                ),


        html.Div(
            className='graph-container',
            children=[
                html.Div(
                    className='graph_height_slider_container',
                    children=[
                        html.Label('Graph Height:', style={'margin-left': '170px'}),
                        dcc.Slider(
                            id='graph-height-slider',
                            min=300,
                            max=700,
                            step=25,
                            value=500,
                            marks={300: '300', 500: '500', 700: '700'},
                            className='graph_height_slider'
                        )
                    ]
                ),
                html.Div(children=[
                    dcc.Graph(id='plot_graph')
                ], className='graph')
            ]
                ),

        html.Div(
            children=[
                dcc.Markdown('Click the photograph icon to download the graph'),
                    ],
            className='not')

                    ]
                )


@app.callback(
    [
        Output('upload-data', 'contents'),
        Output('database-data', 'value'),
        Output('clear-button', 'n_clicks'),
        Output('select_plot', 'value')
    ],
    [
        Input('upload-data', 'contents'),
        Input('database-data', 'value'),
        Input('clear-button', 'n_clicks'),
        Input('select_plot', 'value')
    ]
)
def clear_input_data(content, content2, n_clicks, plot_type):
    if n_clicks > 0:
        return None, None, 0, None
    else:
        return content, content2, 0, plot_type


@app.callback(
    Output('column-list', 'children'),
    Output('row-count-display', 'children'),
    Input('upload-data', 'contents'),
    Input('database-data', 'value')
)
def update_information(contents, content2):
    if contents is None and content2 is None:
        return [], None
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    elif content2 is not None:
        if content2 == 'USD-TRY':
            df = pd.read_csv('data/USDTRY=X.csv')
        elif content2 == 'Car Market':
            df = pd.read_csv('data/turkey_car_market.csv')
        elif content2 == 'Laptop Price':
            df = pd.read_csv('data/laptopPrice.csv')
        elif content2 == 'World Population':
            df = pd.read_csv('data/population.csv')
        elif content2 == 'Air Index':
            df = pd.read_csv('data/air_index.csv')

    columns = df.columns.tolist()
    column_text = " - ".join(columns)
    column_items = html.Span(column_text)
    row_count = len(df)

    return column_items, row_count


@app.callback(Output('header-dropdown', 'options'),
              Input('upload-data', 'contents'),
              Input('database-data', 'value'),
              State('upload-data', 'filename'))
def update_dropdown_1(contents, content2, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            headers = [{'label': h, 'value': h} for h in df.columns]

            return headers
        except Exception as e:
            print(e)
            return []

    elif content2 is not None:
        try:
            if content2 == 'USD-TRY':
                df = pd.read_csv('data/USDTRY=X.csv')
            elif content2 == 'Car Market':
                df = pd.read_csv('data/turkey_car_market.csv')
            elif content2 == 'Laptop Price':
                df = pd.read_csv('data/laptopPrice.csv')
            elif content2 == 'World Population':
                df = pd.read_csv('data/population.csv')
            elif content2 == 'Air Index':
                df = pd.read_csv('data/air_index.csv')
            # Get the headers/column names from the DataFrame
            headers = [{'label': h, 'value': h} for h in df.columns]

            return headers
        except Exception as e:
            print(e)
            return []

    else:
        raise PreventUpdate


@app.callback(Output('header-dropdown-2', 'options'),
              Input('header-dropdown', 'value'),
              Input('upload-data', 'contents'),
              Input('database-data', 'value'),
              State('upload-data', 'filename'))
def update_dropdown_2(selected_header, contents, content2, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            headers = [h for h in df.columns if h != selected_header]
            options = [{'label': h, 'value': h} for h in headers]

            return options

        except Exception as e:
            print(e)
            return []

    elif content2 is not None:
        try:
            if content2 == 'USD-TRY':
                df = pd.read_csv('data/USDTRY=X.csv')
            elif content2 == 'Car Market':
                df = pd.read_csv('data/turkey_car_market.csv')
            elif content2 == 'Laptop Price':
                df = pd.read_csv('data/laptopPrice.csv')
            elif content2 == 'World Population':
                df = pd.read_csv('data/population.csv')
            elif content2 == 'Air Index':
                df = pd.read_csv('data/air_index.csv')
            headers = [h for h in df.columns if h != selected_header]
            options = [{'label': h, 'value': h} for h in headers]

            return options
        except Exception as e:
            print(e)
            return []

    else:
        raise PreventUpdate


@app.callback(Output('header-dropdown-3', 'options'),
              Input('header-dropdown-2', 'value'),
              Input('header-dropdown', 'value'),
              Input('upload-data', 'contents'),
              Input('database-data', 'value'),
              State('upload-data', 'filename'))
def update_dropdown_3(selected_header1, selected_header2, contents, content2, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            headers = [h for h in df.columns if h != selected_header1 and h != selected_header2]
            options = [{'label': h, 'value': h} for h in headers]

            return options

        except Exception as e:
            print(e)
            return []
    elif content2 is not None:
        try:
            if content2 == 'USD-TRY':
                df = pd.read_csv('data/USDTRY=X.csv')
            elif content2 == 'Car Market':
                df = pd.read_csv('data/turkey_car_market.csv')
            elif content2 == 'Laptop Price':
                df = pd.read_csv('data/laptopPrice.csv')
            elif content2 == 'World Population':
                df = pd.read_csv('data/population.csv')
            elif content2 == 'Air Index':
                df = pd.read_csv('data/air_index.csv')
            headers = [h for h in df.columns if h != selected_header1 and h != selected_header2]
            options = [{'label': h, 'value': h} for h in headers]

            return options
        except Exception as e:
            print(e)
            return []

    else:
        raise PreventUpdate


@app.callback(
    Output('row-count', 'min'),
    Output('row-count', 'max'),
    Output('row-count', 'marks'),
    Input('upload-data', 'contents'),
    Input('database-data', 'value'),
    State('upload-data', 'filename')
)
def update_row_count(contents, content2, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            row_count = df.shape[0]
            return 0, row_count, {0: '0', row_count: str(row_count)}

        except Exception as e:
            print(e)
            return 0, 100, {0: '0', 100: '100'}

    elif content2 is not None:
        try:
            if content2 == 'USD-TRY':
                df = pd.read_csv('data/USDTRY=X.csv')
            elif content2 == 'Car Market':
                df = pd.read_csv('data/turkey_car_market.csv')
            elif content2 == 'Laptop Price':
                df = pd.read_csv('data/laptopPrice.csv')
            elif content2 == 'World Population':
                df = pd.read_csv('data/population.csv')
            elif content2 == 'Air Index':
                df = pd.read_csv('data/air_index.csv')

            row_count = df.shape[0]
            return 0, row_count, {0: '0', row_count: str(row_count)}

        except Exception as e:
            print(e)
            return 0, 100, {0: '0', 100: '100'}

    else:
        return 0, 100, {0: '0', 100: '100'}


@app.callback(
    Output('slider-output-container', 'children'),
    Input('row-count', 'value'))
def update_output(value):
    return f"{value[0]} - {value[1]}"


@app.callback([Output('plot_graph', 'figure'), Output("alert-output", "children")],
              State('upload-data', 'contents'),
              State('database-data', 'value'),
              State('select_plot', 'value'),
              State('header-dropdown', 'value'),
              State('header-dropdown-2', 'value'),
              State('header-dropdown-3', 'value'),
              State('color-picker', 'value'),
              State('row-count', 'value'),
              State('random-number', 'value'),
              State('graph-height-slider', 'value'),
              State('graph-width-slider', 'value'),
              Input('submit_button', 'n_clicks'),
              State('upload-data', 'filename'))
def update_scatter_plot(contents, content2, plot_type, value_1, value_2, value_3, color, rows, random_number,
                        height, width, n_clicks, filename):
    if n_clicks == 0:
        fig = go.Figure()
        return fig, html.Div(f"Number of clicks: {n_clicks}")

    if 0 < n_clicks <= 14:
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        elif content2 is not None:
            if content2 == 'USD-TRY':
                df = pd.read_csv('data/USDTRY=X.csv')
            elif content2 == 'Car Market':
                df = pd.read_csv('data/turkey_car_market.csv')
            elif content2 == 'Laptop Price':
                df = pd.read_csv('data/laptopPrice.csv')
            elif content2 == 'World Population':
                df = pd.read_csv('data/population.csv')
            elif content2 == 'Air Index':
                df = pd.read_csv('data/air_index.csv')

        if isinstance(color, dict):
            color = color['hex']

        if value_1 is None or value_2 is None or plot_type is None:
            fig = go.Figure()
            return fig, html.Div(f"Number of clicks: {n_clicks}")

        if random_number is not None:
            df = df.sample(random_number)
        else:
            df = df.iloc[rows[0]:rows[1]+1]

        if value_3:
            if plot_type == 'Scatter Plot':
                fig = px.scatter_3d(df,
                                    x=df[value_1],
                                    y=df[value_2],
                                    z=df[value_3],
                                    color_discrete_sequence=[color]
                                    )

            elif plot_type == 'Line Plot':
                fig = px.line_3d(df,
                                 x=df[value_1],
                                 y=df[value_2],
                                 z=df[value_3],
                                 color_discrete_sequence=[color],
                                 labels={value_1: value_1, value_2: value_2}
                                 )

            elif plot_type == 'Box Plot':
                pass

            elif plot_type == 'Bar Plot':
                pass

            fig.update_layout(xaxis_title=value_1,
                              yaxis_title=value_2,
                              height=height,
                              width=width
                              )

        elif not value_3:
            if plot_type == 'Scatter Plot':
                fig = px.scatter(df,
                                 x=df[value_1],
                                 y=df[value_2],
                                 color_discrete_sequence=[color]
                                 )

            elif plot_type == 'Line Plot':
                fig = px.line(df,
                              x=df[value_1],
                              y=df[value_2],
                              color_discrete_sequence=[color],
                              labels={value_1: value_1, value_2: value_2}
                              )

            elif plot_type == 'Box Plot':
                fig = px.box(df,
                             x=df[value_1],
                             y=df[value_2],
                             color_discrete_sequence=[color],
                             labels={value_1: value_1, value_2: value_2}
                             )

            elif plot_type == 'Bar Plot':
                fig = px.bar(df,
                             x=df[value_1],
                             y=df[value_2],
                             color_discrete_sequence=[color],
                             labels={value_1: value_1, value_2: value_2}
                             )

            fig.update_layout(xaxis_title=value_1,
                              yaxis_title=value_2,
                              height=height,
                              width=width

                              )
        return fig, html.Div(f"Number of clicks: {n_clicks}")

    elif n_clicks > 14:
        fig = go.Figure()
        login_error = html.Div(
            [
                html.H3("Please log in to continue.", style={'color': 'red'})
            ]
        )
        return fig, login_error


@app.callback(Output('output-data-upload', 'children'),
              Output('statistic_button', 'n_clicks'),
              Input('upload-data', 'contents'),
              Input('database-data', 'value'),
              Input('statistic_button', 'n_clicks'),
              State('upload-data', 'filename'))
def update_output(contents, content2, n_clicks, filename):

    if n_clicks == 0:
        return None, n_clicks

    else:

        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            except Exception as e:
                print(e)
                return html.Div(['File can  not loaded.']), 0

        elif content2 is not None:
            try:
                if content2 == 'USD-TRY':
                    df = pd.read_csv('data/USDTRY=X.csv')
                elif content2 == 'Car Market':
                    df = pd.read_csv('data/turkey_car_market.csv')
                elif content2 == 'Laptop Price':
                    df = pd.read_csv('data/laptopPrice.csv')
                elif content2 == 'World Population':
                    df = pd.read_csv('data/population.csv')
                elif content2 == 'Air Index':
                    df = pd.read_csv('data/air_index.csv')

            except Exception as e:
                print(e)

        num_cols = df.select_dtypes(include=np.number).columns

        df_desc = df[num_cols].describe(include='all').round(2).reset_index()
        df_desc.rename(columns={'index': ''}, inplace=True)

        table = html.Div([
            html.H4('Summary Statistics'),
            html.Table(
                # header
                [html.Tr([html.Th(col) for col in df_desc.columns])] +
                # body
                [html.Tr([
                    html.Td(df_desc.iloc[i][col]) for col in df_desc.columns
                ]) for i in range(len(df_desc))]
            )
        ])

        return table, 0


if __name__ == '__main__':
    app.run_server(debug=True)
