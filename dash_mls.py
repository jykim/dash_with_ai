import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import numpy as np
import yaml
from pandas.api.types import is_numeric_dtype
from scipy import stats
from typing import Dict, List, Tuple, Optional
import argparse

class DataLoader:
    """Handles loading and preprocessing of datasets."""
    
    @staticmethod
    def is_binary_variable(series: pd.Series) -> bool:
        """Check if a series is binary (has exactly two unique values)."""
        return len(series.unique()) == 2
    
    @staticmethod
    def convert_to_binary(series: pd.Series) -> Tuple[pd.Series, Optional[str]]:
        """Convert a binary series to 0/1 encoding and return the value mapped to 1."""
        if DataLoader.is_binary_variable(series):
            unique_vals = sorted(series.unique())
            series = series.map({unique_vals[0]: 0, unique_vals[1]: 1})
            return series, unique_vals[1]
        return series, None
    
    @staticmethod
    def load_datasets(config: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Dict]]]:
        """Load all datasets from config and process them."""
        datasets = {}
        numeric_predictors = {}
        
        for dataset_name, dataset_config in config['datasets'].items():
            # Load the dataset
            df = pd.read_csv(dataset_config['file'])
            
            # Convert dependent variable to numeric
            df[dataset_config['dependent_var']] = pd.to_numeric(
                df[dataset_config['dependent_var']], errors='coerce'
            )
            
            # Process predictors
            all_predictors = [p['name'] for p in dataset_config['predictors']]
            numeric_pred = []
            binary_mappings = {}
            
            for col in all_predictors:
                if is_numeric_dtype(df[col]):
                    # Fill missing values with mean for numeric columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean())
                    numeric_pred.append(col)
                elif DataLoader.is_binary_variable(df[col]):
                    converted_series, value_for_one = DataLoader.convert_to_binary(df[col])
                    df[col] = converted_series
                    numeric_pred.append(col)
                    binary_mappings[col] = value_for_one
            
            # Store processed data
            numeric_predictors[dataset_name] = [
                {
                    'name': p['name'],
                    'label': f"{p['label']} ({binary_mappings[p['name']]}=1)" 
                            if p['name'] in binary_mappings else p['label']
                }
                for p in dataset_config['predictors']
                if p['name'] in numeric_pred
            ]
            
            datasets[dataset_name] = df
            
        return datasets, numeric_predictors

class ModelAnalyzer:
    """Handles regression analysis and statistical calculations."""
    
    @staticmethod
    def calculate_anova_results(
        df: pd.DataFrame,
        y: pd.Series,
        selected_predictors: List[str],
        predictor_labels: Dict[str, str]
    ) -> List[Dict]:
        """Calculate Type I ANOVA results for each predictor."""
        total_ss = np.sum((y - y.mean()) ** 2)
        anova_results = []
        residual_ss_prev = total_ss
        n = len(y)
        
        for i in range(len(selected_predictors)):
            current_predictors = selected_predictors[:i+1]
            current_X = sm.add_constant(df[current_predictors].astype(float))
            current_model = sm.OLS(y, current_X).fit()
            
            # Calculate ANOVA metrics
            residual_ss = np.sum(current_model.resid ** 2)
            predictor_ss = residual_ss_prev - residual_ss
            predictor_df = 1
            residual_df = n - (i + 2)
            
            predictor_ms = predictor_ss / predictor_df
            residual_ms = residual_ss / residual_df
            f_stat = predictor_ms / residual_ms
            p_value = 1 - stats.f.cdf(f_stat, predictor_df, residual_df)
            
            anova_results.append({
                'predictor': predictor_labels[selected_predictors[i]],
                'ss': predictor_ss,
                'df': predictor_df,
                'ms': predictor_ms,
                'f_stat': f_stat,
                'p_value': p_value
            })
            
            residual_ss_prev = residual_ss
            
        return anova_results
    
    @staticmethod
    def format_anova_table(results: List[Dict]) -> str:
        """Format ANOVA results into a table string."""
        table = "Type I (Sequential) ANOVA Table:\n\n"
        table += "{:<30} {:>10} {:>8} {:>12} {:>12} {:>10}\n".format(
            "Predictor", "Sum Sq", "DF", "Mean Sq", "F-value", "Pr(>F)"
        )
        table += "-" * 82 + "\n"
        
        for result in results:
            table += "{:<30} {:>10.2f} {:>8d} {:>12.2f} {:>12.2f} {:>10.4f}\n".format(
                result['predictor'],
                result['ss'],
                result['df'],
                result['ms'],
                result['f_stat'],
                result['p_value']
            )
        
        return table

class PlotGenerator:
    """Handles generation of various plots."""
    
    @staticmethod
    def create_residual_plot(
        fitted_values: pd.Series,
        residuals: pd.Series,
        labels: pd.Series,
        dataset_description: str
    ) -> go.Figure:
        """Create residual plot."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fitted_values,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            text=labels,
            hovertemplate=f"<b>Label: %{{text}}</b><br>" +
                         "Fitted value: %{x:.2f}<br>" +
                         "Residual: %{y:.2f}<br>" +
                         "<extra></extra>",
            name='Residuals'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Fitted values",
            yaxis_title="Residuals",
            showlegend=False,
            hovermode='closest',
            height=400,
            margin=dict(t=10, b=40, l=40, r=10)
        )
        return fig
    
    @staticmethod
    def create_correlation_matrix(
        corr_matrix: pd.DataFrame,
        label_names: List[str]
    ) -> go.Figure:
        """Create correlation matrix heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=label_names,
            y=label_names,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdBu',
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            height=400,
            xaxis={'tickangle': 45},
            yaxis={'autorange': 'reversed'},
            margin=dict(t=10, b=40, l=40, r=10)
        )
        return fig

class Dashboard:
    """Main dashboard class that coordinates all components."""
    
    def __init__(self):
        self.config = self._load_config()
        self.tooltips = self._load_tooltips()
        self.datasets, self.numeric_predictors = DataLoader.load_datasets(self.config)
        self.app = self._create_app()
    
    @staticmethod
    def _load_config() -> Dict:
        """Load configuration from YAML file."""
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    @staticmethod
    def _load_tooltips() -> Dict:
        """Load tooltips from YAML file."""
        with open('tooltip.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def _create_app(self) -> dash.Dash:
        """Create and configure the Dash application."""
        app = dash.Dash(__name__)
        
        # Add Font Awesome CSS
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        app.layout = self._create_layout()
        self._setup_callbacks(app)
        
        return app
    
    def _create_layout(self) -> html.Div:
        """Create the dashboard layout."""
        return html.Div([
            html.Div([
                html.H2("Interactive Regression Model Dashboard", style={'fontSize': '28px', 'marginBottom': '20px'}),
                html.I(className="fas fa-question-circle", 
                      title=self.tooltips['dashboard_title'],
                      style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '20px'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            # Controls container
            html.Div([
                # Left side: Dataset selector
                html.Div([
                    html.Div([
                        html.Label("Select Dataset:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                        html.I(className="fas fa-question-circle", 
                              title=self.tooltips['dataset_selector'],
                              style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[{'label': self.config['datasets'][name]['description'], 
                                 'value': name} for name in self.config['datasets'].keys()],
                        value=list(self.config['datasets'].keys())[0],
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),
                
                # Right side: Predictors checklist
                html.Div([
                    html.Div([
                        html.Label("Select Predictors:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                        html.I(className="fas fa-question-circle", 
                              title=self.tooltips['predictors_checklist'],
                              style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    dcc.Checklist(
                        id='predictors-checklist',
                        options=[],
                        value=[],
                        inline=True,
                        style={'maxHeight': '100px', 'overflowY': 'auto'}
                    )
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'marginBottom': '20px'}),
            
            # Results container
            html.Div([
                # Left side: Model summary and ANOVA
                html.Div([
                    html.Div([
                        html.H4("Model Summary", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                        html.I(className="fas fa-question-circle", 
                              title=self.tooltips['model_summary'],
                              style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div(id='model-summary-div', style={
                        'whiteSpace': 'pre-wrap', 
                        'fontFamily': 'monospace',
                        'marginTop': '20px'
                    }),
                    html.Hr(style={'margin': '20px 0'}),
                    html.Div([
                        html.H4("Type I ANOVA Analysis", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                        html.I(className="fas fa-question-circle", 
                              title=self.tooltips['anova_analysis'],
                              style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div(id='anova-table-div', style={
                        'whiteSpace': 'pre-wrap',
                        'fontFamily': 'monospace'
                    })
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right side: Plots
                html.Div([
                    html.Div([
                        html.H4("Residual Plot", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                        html.I(className="fas fa-question-circle", 
                              title=self.tooltips['residual_plot'],
                              style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    dcc.Graph(id='residual-plot'),
                    html.Div([
                        html.H4("Correlation Matrix", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                        html.I(className="fas fa-question-circle", 
                              title=self.tooltips['correlation_matrix'],
                              style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    dcc.Graph(id='correlation-matrix')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Data table
            html.Hr(),
            html.Div([
                html.H4("Interactive Data Table", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.I(className="fas fa-question-circle", 
                      title=self.tooltips['data_table'],
                      style={'marginLeft': '10px', 'cursor': 'help', 'color': '#007bff', 'fontSize': '16px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
            dash_table.DataTable(
                id='data-table',
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                row_selectable="multi",
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
                style_table={
                    'maxHeight': '400px',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'width': '90%',
                    'margin': '0 auto'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'minWidth': '100px',
                    'maxWidth': '180px',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold',
                    'border': '1px solid black'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ])
    
    def _setup_callbacks(self, app: dash.Dash):
        """Set up all dashboard callbacks."""
        
        @app.callback(
            [Output('predictors-checklist', 'options'),
             Output('predictors-checklist', 'value'),
             Output('data-table', 'columns'),
             Output('data-table', 'data')],
            [Input('dataset-selector', 'value')]
        )
        def update_dataset_elements(selected_dataset):
            predictor_options = [
                {'label': p['label'], 'value': p['name']} 
                for p in self.numeric_predictors[selected_dataset]
            ]
            predictor_values = [p['name'] for p in self.numeric_predictors[selected_dataset]]
            
            df = self.datasets[selected_dataset]
            columns = [{"name": col.capitalize(), "id": col, "selectable": True} 
                      for col in df.columns]
            
            return predictor_options, predictor_values, columns, df.to_dict('records')
        
        @app.callback(
            [Output('model-summary-div', 'children'),
             Output('anova-table-div', 'children'),
             Output('residual-plot', 'figure'),
             Output('correlation-matrix', 'figure')],
            [Input('dataset-selector', 'value'),
             Input('predictors-checklist', 'value')]
        )
        def update_regression_model(selected_dataset, selected_predictors):
            if not selected_predictors:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="No predictors selected",
                    xaxis_title="Fitted values",
                    yaxis_title="Residuals"
                )
                return "No predictors selected. Please select at least one.", "", empty_fig, empty_fig
            
            try:
                df = self.datasets[selected_dataset]
                dataset_config = self.config['datasets'][selected_dataset]
                
                # Prepare data
                X = df[selected_predictors].astype(float)
                y = df[dataset_config['dependent_var']].astype(float)
                predictor_labels = {p['name']: p['label'] 
                                  for p in self.numeric_predictors[selected_dataset]}
                
                # Calculate ANOVA results
                anova_results = ModelAnalyzer.calculate_anova_results(
                    df, y, selected_predictors, predictor_labels
                )
                anova_table = ModelAnalyzer.format_anova_table(anova_results)
                
                # Fit final model
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                
                # Create plots
                residual_fig = PlotGenerator.create_residual_plot(
                    model.fittedvalues,
                    model.resid,
                    df[dataset_config['label_var']],
                    dataset_config['description']
                )
                
                corr_data = pd.concat([y, X], axis=1)
                corr_matrix = corr_data.corr()
                label_names = [dataset_config['dependent_var'].capitalize()] + \
                            [predictor_labels[p] for p in selected_predictors]
                
                corr_fig = PlotGenerator.create_correlation_matrix(
                    corr_matrix, label_names
                )
                
                return model.summary().as_text(), anova_table, residual_fig, corr_fig
                
            except Exception as e:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Error in regression",
                    xaxis_title="Fitted values",
                    yaxis_title="Residuals"
                )
                return f"Error in regression: {str(e)}", "", empty_fig, empty_fig
    
    def run(self, port=8050):
        """Run the dashboard."""
        self.app.run_server(debug=True, port=port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the dashboard')
    parser.add_argument('--port', type=int, default=8052, help='Port to run the server on')
    args = parser.parse_args()
    
    dashboard = Dashboard()
    dashboard.run(port=args.port)

# Create the app instance at module level for WSGI servers
dashboard = Dashboard()
app = dashboard.app
# Fixed using WSGI servers:
# https://www.pythonanywhere.com/forums/topic/11625/#id_post_43420
