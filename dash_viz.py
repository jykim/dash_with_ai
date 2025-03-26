import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import yaml
from pandas.api.types import is_numeric_dtype
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

class VisualizationDashboard:
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
            html.H2("Data Visualization Dashboard", style={'textAlign': 'center'}),
            
            # Controls container
            html.Div([
                # Left side: Dataset selector
                html.Div([
                    html.Label("Select Dataset:"),
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[{'label': self.config['datasets'][name]['description'], 
                                'value': name} for name in self.config['datasets'].keys()],
                        value=list(self.config['datasets'].keys())[0]
                    )
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
                # Scatterplot matrix
                html.Div([
                    html.H4("Scatterplot Matrix", style={'textAlign': 'center'}),
                    dcc.Graph(id='scatterplot-matrix', style={'height': '800px'})
                ], style={'marginBottom': '20px'}),
                
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
    
    def _create_scatterplot_matrix(self, df, selected_variables, selected_dataset):
        """Create an nxn scatterplot matrix for selected variables."""
        if not selected_variables or len(selected_variables) < 2:
            return go.Figure()
        
        # Create subplots
        n_vars = len(selected_variables)
        fig = make_subplots(
            rows=n_vars, cols=n_vars,
            subplot_titles=[f"{var}" for var in selected_variables],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        # Get variable labels from config
        var_labels = {}
        dep_var = self.config['datasets'][selected_dataset]['dependent_var']
        label_var = self.config['datasets'][selected_dataset]['label_var']
        
        # Check if label column exists
        has_label = label_var in df.columns
        
        for var in selected_variables:
            # Check if it's the dependent variable
            if var == dep_var:
                var_labels[var] = f"{var} (Dependent)"
            else:
                # Look for the label in predictors
                for p in self.numeric_predictors[selected_dataset]:
                    if p['name'] == var:
                        var_labels[var] = p['label']
                        break
        
        # Add scatterplots
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Determine which variable is the dependent variable
                    y_var = selected_variables[i]
                    x_var = selected_variables[j]
                    
                    # Create hover text based on whether label column exists
                    if has_label:
                        hover_text = (
                            f"<b>{label_var}: %{{customdata}}</b><br>" +
                            f"<b>{var_labels.get(y_var, y_var)}</b><br>" +
                            f"<b>{var_labels.get(x_var, x_var)}</b><br>" +
                            f"x: %{{x:.2f}}<br>" +
                            f"y: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df[x_var],
                                y=df[y_var],
                                mode='markers',
                                marker=dict(size=5, opacity=0.5),
                                showlegend=False,
                                hovertemplate=hover_text,
                                customdata=df[label_var]  # Add label data for hover
                            ),
                            row=i+1, col=j+1
                        )
                    else:
                        hover_text = (
                            f"<b>{var_labels.get(y_var, y_var)}</b><br>" +
                            f"<b>{var_labels.get(x_var, x_var)}</b><br>" +
                            f"x: %{{x:.2f}}<br>" +
                            f"y: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df[x_var],
                                y=df[y_var],
                                mode='markers',
                                marker=dict(size=5, opacity=0.5),
                                showlegend=False,
                                hovertemplate=hover_text
                            ),
                            row=i+1, col=j+1
                        )
                else:
                    # Add histogram on diagonal
                    var = selected_variables[i]
                    hover_text = (
                        f"<b>{var_labels.get(var, var)}</b><br>" +
                        f"Value: %{{x:.2f}}<br>" +
                        f"Count: %{{y}}<br>" +
                        "<extra></extra>"
                    )
                    
                    fig.add_trace(
                        go.Histogram(
                            x=df[var],
                            name=var,
                            showlegend=False,
                            hovertemplate=hover_text
                        ),
                        row=i+1, col=j+1
                    )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Scatterplot Matrix",
            title_x=0.5,
            title_y=0.95
        )
        
        # Update axes labels
        for i in range(n_vars):
            fig.update_xaxes(title_text=var_labels.get(selected_variables[i], selected_variables[i]), row=i+1, col=n_vars)
            fig.update_yaxes(title_text=var_labels.get(selected_variables[i], selected_variables[i]), row=1, col=i+1)
        
        return fig

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
            Output('scatterplot-matrix', 'figure'),
            [Input('dataset-selector', 'value'),
             Input('variables-checklist', 'value')]
        )
        def update_scatterplot_matrix(selected_dataset, selected_variables):
            if not selected_variables or len(selected_variables) < 2:
                return go.Figure()
            
            df = self.datasets[selected_dataset]
            return self._create_scatterplot_matrix(df, selected_variables, selected_dataset)
    
    def run(self, port=8050):
        """Run the dashboard."""
        self.app.run_server(debug=True, port=port)

if __name__ == '__main__':
    dashboard = VisualizationDashboard()
    dashboard.run(port=8052)

# Create the app instance at module level for WSGI servers
dashboard = VisualizationDashboard()
app = dashboard.app 