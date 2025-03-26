import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import yaml
from pandas.api.types import is_numeric_dtype
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.GraphUtils import GraphUtils
import base64
from io import BytesIO
import plotly.graph_objects as go
import tempfile
import os

class DataLoader:
    """Handles loading and preprocessing of datasets."""
    
    @staticmethod
    def is_binary_variable(series: pd.Series) -> bool:
        """Check if a series is binary (has exactly two unique values)."""
        return len(series.unique()) == 2
    
    @staticmethod
    def convert_to_binary(series: pd.Series):
        """Convert a binary series to 0/1 encoding."""
        if DataLoader.is_binary_variable(series):
            unique_vals = sorted(series.unique())
            return series.map({unique_vals[0]: 0, unique_vals[1]: 1})
        return series
    
    @staticmethod
    def load_datasets(config):
        """Load all datasets from config and process them."""
        datasets = {}
        numeric_predictors = {}
        
        for dataset_name, dataset_config in config['datasets'].items():
            # Load the dataset
            df = pd.read_csv(dataset_config['file'])
            
            # Process variables
            all_vars = [p['name'] for p in dataset_config['predictors']]
            if 'dependent_var' in dataset_config:
                all_vars.append(dataset_config['dependent_var'])
            
            numeric_vars = []
            
            for col in all_vars:
                if is_numeric_dtype(df[col]):
                    # Fill missing values with mean for numeric columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean())
                    numeric_vars.append(col)
                elif DataLoader.is_binary_variable(df[col]):
                    df[col] = DataLoader.convert_to_binary(df[col])
                    numeric_vars.append(col)
            
            # Store processed data
            numeric_predictors[dataset_name] = [
                {
                    'name': p['name'],
                    'label': p['label']
                }
                for p in dataset_config['predictors']
                if p['name'] in numeric_vars
            ]
            
            datasets[dataset_name] = df
            
        return datasets, numeric_predictors

class CausalDiscovery:
    """Handles causal discovery analysis."""
    
    @staticmethod
    def run_algorithm(data, algorithm='pc', alpha=0.05, stable=True):
        """Run selected causal discovery algorithm on the data."""
        try:
            # Convert data to numpy array and prepare labels
            data_np = data.values
            labels = [f'{col}' for col in data.columns]
            
            # Run selected algorithm
            if algorithm == 'pc':
                cg = pc(data_np, alpha=alpha, stable=stable)
                graph = cg.G
            elif algorithm == 'fci':
                cg, edges = fci(data_np, alpha=alpha)
                graph = cg
            elif algorithm == 'ges':
                record = ges(data_np)
                graph = record['G']
            elif algorithm == 'grasp':
                graph = grasp(data_np)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Create a temporary file for the PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Convert to PNG using the temporary file with labels
            pyd = GraphUtils.to_pydot(graph, labels=labels)
            pyd.write_png(temp_path)
            
            # Read the file and encode to base64
            with open(temp_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode()
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return encoded_image, None
            
        except Exception as e:
            return None, str(e)

class Dashboard:
    """Main dashboard class that coordinates all components."""
    
    def __init__(self):
        self.config = self._load_config()
        self.datasets, self.numeric_predictors = DataLoader.load_datasets(self.config)
        self.app = self._create_app()
    
    @staticmethod
    def _load_config():
        """Load configuration from YAML file."""
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def _create_app(self):
        """Create and configure the Dash application."""
        app = dash.Dash(__name__)
        
        app.layout = self._create_layout()
        self._setup_callbacks(app)
        
        return app
    
    def _create_layout(self):
        """Create the dashboard layout."""
        return html.Div([
            html.H2("Causal Discovery Dashboard", style={'textAlign': 'center'}),
            
            # Controls container
            html.Div([
                # Left side: Dataset and Algorithm selectors
                html.Div([
                    # Dataset selector
                    html.Div([
                        html.Label("Select Dataset:"),
                        dcc.Dropdown(
                            id='dataset-selector',
                            options=[{'label': self.config['datasets'][name]['description'], 
                                    'value': name} for name in self.config['datasets'].keys()],
                            value=list(self.config['datasets'].keys())[0]
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Model selector
                    html.Div([
                        html.Label("Select Algorithm:"),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': 'PC Algorithm', 'value': 'pc'},
                                {'label': 'FCI Algorithm', 'value': 'fci'},
                                {'label': 'GES Algorithm', 'value': 'ges'},
                                {'label': 'GRaSP Algorithm', 'value': 'grasp'}
                            ],
                            value='ges'
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Algorithm parameters
                    html.Div([
                        html.Label("Significance Level (α):"),
                        dcc.Slider(
                            id='alpha-slider',
                            min=0.01,
                            max=0.1,
                            step=0.01,
                            value=0.05,
                            marks={i/100: str(i/100) for i in range(1, 11)}
                        )
                    ])
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),
                
                # Right side: Variables selector
                html.Div([
                    html.Label("Select Variables:"),
                    dcc.Checklist(
                        id='variables-checklist',
                        options=[],
                        value=[],
                        style={
                            'maxHeight': '300px',
                            'overflowY': 'auto',
                            'padding': '10px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '5px'
                        }
                    )
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            # Results container
            html.Div([
                # Side-by-side container for graph and info
                html.Div([
                    # Left side: Causal Graph
                    html.Div([
                        html.H4("Causal Graph", style={'textAlign': 'center'}),
                        html.Img(id='causal-graph', style={'maxWidth': '100%'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    # Right side: Algorithm Explanation
                    html.Div([
                        html.H4("Algorithm Information", style={'textAlign': 'center'}),
                        html.Div(id='algorithm-info', style={
                            'padding': '20px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '5px',
                            'marginTop': '20px',
                            'height': '100%',
                            'overflowY': 'auto'
                        })
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
                
                # Error messages
                html.Div(id='error-message', style={'color': 'red', 'marginTop': '10px'}),
                
                # Data table
                html.H4("Data Preview", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='data-table',
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ])
        ])
    
    def _get_algorithm_info(self, algorithm, alpha):
        """Get information about the selected algorithm."""
        info = {
            'pc': {
                'name': 'PC Algorithm',
                'description': 'A constraint-based algorithm that uses conditional independence tests to learn the causal structure.',
                'parameters': f'Significance level (α): {alpha}',
                'assumptions': [
                    'Causal faithfulness',
                    'No unobserved confounders',
                    'No selection bias'
                ],
                'strengths': [
                    'Fast and efficient',
                    'Can handle high-dimensional data',
                    'Provides confidence in edge directions'
                ],
                'limitations': [
                    'Sensitive to violations of assumptions',
                    'May miss some causal relationships',
                    'Requires large sample sizes'
                ]
            },
            'fci': {
                'name': 'FCI Algorithm',
                'description': 'A constraint-based algorithm that can handle unobserved confounders and selection bias.',
                'parameters': f'Significance level (α): {alpha}',
                'assumptions': [
                    'Causal faithfulness',
                    'No selection bias'
                ],
                'strengths': [
                    'Can handle unobserved confounders',
                    'More robust than PC',
                    'Provides partial ancestral graphs'
                ],
                'limitations': [
                    'More computationally intensive',
                    'May be less informative than PC',
                    'Requires larger sample sizes'
                ]
            },
            'ges': {
                'name': 'GES Algorithm',
                'description': 'A score-based algorithm that searches for the best Bayesian network structure.',
                'parameters': 'No significance level parameter',
                'assumptions': [
                    'Multivariate normal distribution',
                    'No unobserved confounders',
                    'No selection bias'
                ],
                'strengths': [
                    'Consistent under faithfulness',
                    'Can handle non-linear relationships',
                    'Provides complete DAGs'
                ],
                'limitations': [
                    'Assumes continuous data',
                    'Computationally intensive',
                    'May not scale well to high dimensions'
                ]
            },
            'grasp': {
                'name': 'GRaSP Algorithm',
                'description': 'A permutation-based algorithm that searches for the sparsest DAG consistent with the data.',
                'parameters': 'No significance level parameter',
                'assumptions': [
                    'No unobserved confounders',
                    'No selection bias'
                ],
                'strengths': [
                    'Finds sparse structures',
                    'Robust to violations of assumptions',
                    'Can handle non-linear relationships'
                ],
                'limitations': [
                    'May be less informative than other methods',
                    'Computationally intensive',
                    'May not scale well to high dimensions'
                ]
            }
        }
        
        algo_info = info[algorithm]
        return html.Div([
            html.H5(algo_info['name'], style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P(algo_info['description'], style={'marginBottom': '15px'}),
            html.H6('Parameters:', style={'color': '#2c3e50', 'marginBottom': '5px'}),
            html.P(algo_info['parameters'], style={'marginBottom': '15px'}),
            html.H6('Assumptions:', style={'color': '#2c3e50', 'marginBottom': '5px'}),
            html.Ul([html.Li(item) for item in algo_info['assumptions']], style={'marginBottom': '15px'}),
            html.H6('Strengths:', style={'color': '#2c3e50', 'marginBottom': '5px'}),
            html.Ul([html.Li(item) for item in algo_info['strengths']], style={'marginBottom': '15px'}),
            html.H6('Limitations:', style={'color': '#2c3e50', 'marginBottom': '5px'}),
            html.Ul([html.Li(item) for item in algo_info['limitations']])
        ])

    def _setup_callbacks(self, app):
        """Set up all dashboard callbacks."""
        
        @app.callback(
            [Output('variables-checklist', 'options'),
             Output('variables-checklist', 'value'),
             Output('data-table', 'columns'),
             Output('data-table', 'data')],
            [Input('dataset-selector', 'value')]
        )
        def update_dataset_elements(selected_dataset):
            # Get all numeric variables including the dependent variable
            all_vars = []
            
            # Add dependent variable if it's numeric
            dep_var = self.config['datasets'][selected_dataset]['dependent_var']
            df = self.datasets[selected_dataset]
            if is_numeric_dtype(df[dep_var]) or DataLoader.is_binary_variable(df[dep_var]):
                all_vars.append({
                    'label': f"{dep_var} (Dependent)",
                    'value': dep_var
                })
            
            # Add predictor variables
            for p in self.numeric_predictors[selected_dataset]:
                all_vars.append({
                    'label': p['label'],
                    'value': p['name']
                })
            
            # Set all variables as selected by default
            selected_values = [var['value'] for var in all_vars]
            
            # Create columns for data table
            columns = [{"name": col.capitalize(), "id": col} for col in df.columns]
            
            return all_vars, selected_values, columns, df.to_dict('records')
        
        @app.callback(
            [Output('causal-graph', 'src'),
             Output('error-message', 'children'),
             Output('algorithm-info', 'children')],
            [Input('dataset-selector', 'value'),
             Input('variables-checklist', 'value'),
             Input('alpha-slider', 'value'),
             Input('model-selector', 'value')]
        )
        def update_causal_graph(selected_dataset, selected_variables, alpha, algorithm):
            if not selected_variables or len(selected_variables) < 2:
                return None, "Please select at least two variables for causal discovery.", None
            
            try:
                df = self.datasets[selected_dataset]
                data_subset = df[selected_variables]
                
                # Run causal discovery
                encoded_image, error = CausalDiscovery.run_algorithm(
                    data_subset, 
                    algorithm=algorithm,
                    alpha=alpha
                )
                
                if error:
                    return None, f"Error in causal discovery: {error}", None
                
                # Get algorithm information
                algorithm_info = self._get_algorithm_info(algorithm, alpha)
                
                return f'data:image/png;base64,{encoded_image}', None, algorithm_info
                
            except Exception as e:
                return None, f"Error: {str(e)}", None
    
    def run(self, port=8050):
        """Run the dashboard."""
        self.app.run_server(debug=True, port=port)

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.run(port=8052)

# Create the app instance at module level for WSGI servers
dashboard = Dashboard()
app = dashboard.app 