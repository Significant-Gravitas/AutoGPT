import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
import os

# Sample data
# Function to load data based on the selected folder
def load_data(folder_name):
    with open(f"./agbenchmark/reports/{folder_name}/report.json", "r") as f:
        return json.load(f)


# List the available subfolders in the reports directory
available_folders = sorted([f for f in os.listdir("./agbenchmark/reports") if os.path.isdir(os.path.join("./agbenchmark/reports", f))])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def generate_table(data_frame):
    return dbc.Table(
        # Header
        [html.Thead(html.Tr([html.Th(col) for col in data_frame.columns]))] +
        # Body
        [html.Tbody([
            html.Tr([
                html.Td(data_frame.iloc[i][col], style={'backgroundColor': '#77dd77' if data_frame.iloc[i]['Status'] == 'Passed' else '#ff6961'}) for col in data_frame.columns
            ]) for i in range(len(data_frame))
        ])]
    )

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AG Benchmark Tests Overview"), width={"size": 6, "offset": 3}),
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id="folder-selector",
                options=[{"label": folder_name, "value": folder_name} for folder_name in available_folders],
                value=None,
                placeholder="Select a folder to load data"
            ),
            html.Div(id="folder-data-output")
        ])
    ]),
])


@app.callback(
    Output("folder-data-output", "children"),
    [Input("folder-selector", "value")]
)
def display_folder_data(selected_folder):
    if not selected_folder:
        return "No folder selected"

    data = load_data(selected_folder)

    # Extract the necessary data from the report
    command = data['command']
    benchmark_git_commit_sha = data['benchmark_git_commit_sha'] or "N/A"
    benchmark_git_commit_sha = benchmark_git_commit_sha.split('/')[-1][:8] if benchmark_git_commit_sha != "N/A" else "N/A"
    agent_git_commit_sha = data['agent_git_commit_sha'] or "N/A"
    agent_git_commit_sha = agent_git_commit_sha.split('/')[-1][:8] if agent_git_commit_sha != "N/A" else "N/A"
    completion_time = data['completion_time']
    benchmark_start_time = data['benchmark_start_time']
    run_time = data['metrics']['run_time']
    highest_difficulty = data['metrics']['highest_difficulty']

    return [
    dbc.Row([
        dbc.Col(html.Div("Start Time: " + benchmark_start_time), width=3),
    
        dbc.Col(html.Div("Run Time: " + run_time), width=3),
        dbc.Col(html.Div("Highest Difficulty Achieved: " + highest_difficulty), width=3),
        dbc.Col(html.Div("Benchmark Git Commit: " + benchmark_git_commit_sha), width=3),
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col(html.Div("Completion Time: " + completion_time), width=3),
        dbc.Col(html.Div("Command: " + command), width=3),    
        dbc.Col(), # Empty column for alignment
        
        dbc.Col(html.Div("Agent Git Commit: " + agent_git_commit_sha), width=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="category-pass-rate"),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            generate_table(pd.DataFrame({
                'Test Name': list(data['tests'].keys()),
                'Status': ['Passed' if t['metrics'].get('success', False) else 'Failed' for t in data['tests'].values()]
            }))
        ])
    ])
    ]

@app.callback(
    Output("subtest-output", "children"),
    [Input("test-selector", "value")]
)
def display_subtests(selected_test):
    if not selected_test:
        return "No test selected"

    subtests = data['tests'][selected_test]['tests']
    df = pd.DataFrame({
        'Subtest Name': list(subtests.keys()),
        'Status': ['Passed' if st['metrics']['success'] else 'Failed' for st in subtests.values()]
    })
    return generate_table(df)

@app.callback(
    Output('category-pass-rate', 'figure'),
    [Input('folder-selector', 'value')]
)
def update_radar_chart(selected_folder):
    if not selected_folder:
        return "No folder selected"

    data = load_data(selected_folder)
    # Extract all categories from the data
    categories = set()
    for test in data['tests'].keys():
        if 'category' not in data['tests'][test]:
            print(f"Test {test} has no category")
            continue
        cat = data['tests'][test]['category']
        categories.update(cat)

    # Calculate pass rates for each category
    pass_rate = {}
    for cat in categories:
        total_tests = 0
        passed_tests = 0
        for test in data['tests'].keys():
            if 'category' not in data['tests'][test] or cat not in data['tests'][test]['category']:
                continue
            total_tests = total_tests + 1 if cat in data['tests'][test]['category'] else total_tests
            passed_tests = passed_tests + 1 if cat in data['tests'][test]['category'] and data['tests'][test]['metrics']['success'] else passed_tests
        pass_rate[cat] = (passed_tests / total_tests) * 100

    df = pd.DataFrame({
        'Category': list(pass_rate.keys()),
        'Pass Rate (%)': list(pass_rate.values())
    }).sort_values(by=['Category'], ascending=True)
    
    fig = px.line_polar(df, r='Pass Rate (%)', theta='Category', line_close=True, template="plotly", title="Pass Rate by Category")
    fig.update_traces(fill='toself')

    # Set the radial axis maximum range to 100
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]  # Setting range from 0 to 100%
            )
        )
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
