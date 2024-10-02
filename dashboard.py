import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px

app = dash.Dash(__name__)
app.title = "Comparative Dashboard"


def calculate_npv(cash_flows, discount_rate):
    return npf.npv(discount_rate, cash_flows)


def calculate_irr(cash_flows):
    return npf.irr(cash_flows)


def calculate_pbp(cash_flows):
    cumulative_cash_flows = np.cumsum(cash_flows)
    return np.argmax(cumulative_cash_flows >= 0)


def calculate_annual_revenue(price, demand):
    return price * demand


def calculate_net_cash_flows(initial_investment, annual_cash_flows, product_lifetime):
    return [-initial_investment] + [annual_cash_flows] * product_lifetime


products = ['Bioethanol', 'CMS', 'E14_12', 'E14_14']

app.layout = html.Div([
    html.H1('Comparative Dashboard', style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.Div([
            html.H3(f'Inputs for {product}', style={'textAlign': 'center'}),
            dcc.Input(id=f'init_investment_{product}', type='number',
                      placeholder=f'Initial Investment ($)', style={'margin': '5px'}),
            dcc.Input(id=f'ann_cash_flows_{product}', type='number',
                      placeholder=f'Annual Cash Flows ($)', style={'margin': '5px'}),
            dcc.Input(id=f'disc_rate_{product}', type='number',
                      placeholder=f'Discount Rate (%)', style={'margin': '5px'}),
            dcc.Input(id=f'prod_lifetime_{product}', type='number',
                      placeholder=f'Product Lifetime (years)', style={'margin': '5px'}),
            dcc.Input(id=f'market_demand_{product}', type='number',
                      placeholder=f'Market Demand', style={'margin': '5px'}),
            dcc.Input(id=f'prod_cost_{product}', type='number',
                      placeholder=f'Production Cost ($)', style={'margin': '5px'}),
            dcc.Input(id=f'sell_price_{product}', type='number',
                      placeholder=f'Selling Price ($)', style={'margin': '5px'})
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px',
                  'margin': '10px', 'width': '300px'}) for product in products
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    html.Div([
        html.Button('Submit', id='submit-button', n_clicks=0,
                    style={'margin': '20px', 'padding': '10px 20px', 'fontSize': '16px'})
    ], style={'textAlign': 'center'}),
    html.Div(id='output-data', style={'textAlign': 'center', 'margin': '20px'}),
    html.Div(id='table-container', style={'margin': '20px'}),
    html.Div([
        dcc.Graph(id='npv-graph', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='irr-graph', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='pbp-graph', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='revenue-graph', style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
])


@app.callback(
    [Output('output-data', 'children'),
     Output('npv-graph', 'figure'),
     Output('irr-graph', 'figure'),
     Output('pbp-graph', 'figure'),
     Output('revenue-graph', 'figure'),
     Output('table-container', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State(f'init_investment_{product}', 'value') for product in products] +
    [State(f'ann_cash_flows_{product}', 'value') for product in products] +
    [State(f'disc_rate_{product}', 'value') for product in products] +
    [State(f'prod_lifetime_{product}', 'value') for product in products] +
    [State(f'market_demand_{product}', 'value') for product in products] +
    [State(f'prod_cost_{product}', 'value') for product in products] +
    [State(f'sell_price_{product}', 'value') for product in products]
)
def update_output(n_clicks, *values):
    data = {}
    for i, product in enumerate(products):
        init_investment = values[i]
        ann_cash_flows = values[i + 4]
        disc_rate = values[i + 8] / 100 if values[i + 8] else None
        prod_lifetime = values[i + 12]
        market_demand = values[i + 16]
        prod_cost = values[i + 20]
        sell_price = values[i + 24]

        if all([init_investment, ann_cash_flows, disc_rate, prod_lifetime, market_demand, prod_cost, sell_price]):
            net_cash_flows = calculate_net_cash_flows(init_investment, ann_cash_flows, prod_lifetime)
            npv = calculate_npv(net_cash_flows, disc_rate)
            irr = calculate_irr(net_cash_flows)
            pbp = calculate_pbp(net_cash_flows)
            annual_revenue = calculate_annual_revenue(sell_price, market_demand)

            data[product] = {
                'NPV': npv,
                'IRR': irr,
                'PBP': pbp,
                'Annual Revenue': annual_revenue,
                'Net Cash Flows': ', '.join(map(str, net_cash_flows))  # Properly separate the numbers
            }

    if data:
        df = pd.DataFrame(data).T
        df['Product'] = df.index
        df = df.reset_index(drop=True)
        npv_fig = px.bar(df, x='Product', y='NPV', title='Net Present Value (NPV)')
        irr_fig = px.bar(df, x='Product', y='IRR', title='Internal Rate of Return (IRR)')
        pbp_fig = px.bar(df, x='Product', y='PBP', title='Payback Period (PBP)')
        revenue_fig = px.bar(df, x='Product', y='Annual Revenue', title='Annual Revenue')

        table = html.Table([
            html.Thead(
                html.Tr([html.Th('Product')] + [html.Th(col) for col in df.columns if col != 'Product'])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df.iloc[i]['Product'])] + [html.Td(df.iloc[i][col]) for col in
                                                       df.columns if col != 'Product']
                ) for i in range(len(df))
            ])
        ], style={
            'width': '100%', 'border': '1px solid black', 'borderCollapse': 'collapse', 'textAlign': 'center',
            'padding': '10px', 'lineHeight': '1.5', 'margin': '10px 0',
            'borderSpacing': '0', 'border': '1px solid black'
        })

        return ('Data successfully entered! Scroll down to see the results.',
                npv_fig, irr_fig, pbp_fig, revenue_fig, table)

    return 'Please fill in all fields.', {}, {}, {}, {}, None


if __name__ == '__main__':
    app.run_server(debug=True)

server = app.server
