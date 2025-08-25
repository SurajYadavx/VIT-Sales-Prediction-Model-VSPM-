from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
import os
from werkzeug.utils import secure_filename
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import csv
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-premium-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Simple CSV reader without pandas
def read_csv_simple(filepath):
    """Read CSV file without pandas"""
    data = []
    headers = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            
            for row in csv_reader:
                data.append(row)
    except:
        return {}, []
    
    # Convert to dict format
    df_dict = {}
    for i, header in enumerate(headers):
        column_data = []
        for row in data:
            try:
                # Try to convert to float
                if i < len(row) and row[i] and row[i].strip():
                    value = float(row[i])
                    column_data.append(value)
                else:
                    column_data.append(0.0)
            except (ValueError, IndexError):
                # Keep as string if conversion fails
                column_data.append(row[i] if i < len(row) else '')
        df_dict[header] = column_data
    
    return df_dict, headers

def get_numeric_columns(df_dict):
    """Get columns that contain numeric data"""
    numeric_cols = []
    for col_name, col_data in df_dict.items():
        # Check if most values are numeric
        numeric_count = 0
        for value in col_data[:min(10, len(col_data))]:  # Check first 10 values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_count += 1
        
        if numeric_count > len(col_data[:10]) * 0.7:  # 70% numeric
            numeric_cols.append(col_name)
    
    return numeric_cols

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df_dict, headers = read_csv_simple(filepath)
                
                if not df_dict:
                    flash('The uploaded file is empty or invalid', 'error')
                    return redirect(request.url)
                
                # Save processed data
                save_processed_data(df_dict, headers)
                flash('File uploaded and processed successfully!', 'success')
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Only CSV files are allowed', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

def save_processed_data(df_dict, headers):
    """Save processed data to CSV"""
    try:
        with open('data/processed_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            
            # Get the length of data
            data_length = len(list(df_dict.values())[0]) if df_dict else 0
            
            for i in range(data_length):
                row = []
                for header in headers:
                    row.append(df_dict[header][i] if i < len(df_dict[header]) else '')
                writer.writerow(row)
    except Exception as e:
        print(f"Error saving data: {e}")

@app.route('/dashboard')
def dashboard():
    try:
        df_dict, headers = read_csv_simple('data/processed_data.csv')
        
        if not df_dict:
            flash('No data found. Please upload a CSV file first.', 'warning')
            return redirect(url_for('upload_file'))
            
        numeric_cols = get_numeric_columns(df_dict)
        
        stats = {
            'total_rows': len(list(df_dict.values())[0]) if df_dict else 0,
            'total_columns': len(headers),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(headers) - len(numeric_cols),
            'columns': headers,
            'numeric_cols': numeric_cols,
            'missing_values': {col: 0 for col in headers},  # Simplified
            'data_types': {col: 'numeric' if col in numeric_cols else 'text' for col in headers},
            'memory_usage': 1024 * len(headers),  # Estimated
            'duplicates': 0
        }
        
        # Generate basic charts
        charts = generate_working_charts(df_dict, numeric_cols)
        
        return render_template('dashboard.html', stats=stats, charts=charts)
        
    except FileNotFoundError:
        flash('No data found. Please upload a CSV file first.', 'warning')
        return redirect(url_for('upload_file'))
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

def generate_working_charts(df_dict, numeric_cols):
    """Generate charts that work without pandas"""
    charts = {}
    
    if len(numeric_cols) > 0:
        try:
            col_name = numeric_cols[0]
            col_data = df_dict[col_name]
            
            # Filter out non-numeric values
            numeric_data = [x for x in col_data if isinstance(x, (int, float))]
            
            if numeric_data:
                # Simple histogram using plotly.graph_objects (no pandas needed)
                fig_dist = go.Figure(data=[go.Histogram(x=numeric_data, name=col_name)])
                fig_dist.update_layout(
                    title=f"Distribution of {col_name}",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(47,47,47,0.1)',
                    font=dict(color='#2F2F2F'),
                    title_font=dict(size=16, color='#DAA520')
                )
                charts['distribution'] = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)
                
                # Simple scatter plot if we have 2+ numeric columns
                if len(numeric_cols) >= 2:
                    col2_name = numeric_cols[1]
                    col2_data = df_dict[col2_name]
                    
                    # Filter both columns for numeric values
                    x_data = []
                    y_data = []
                    for i in range(min(len(numeric_data), len(col2_data))):
                        if isinstance(col_data[i], (int, float)) and isinstance(col2_data[i], (int, float)):
                            x_data.append(col_data[i])
                            y_data.append(col2_data[i])
                    
                    if x_data and y_data:
                        fig_scatter = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='markers'))
                        fig_scatter.update_layout(
                            title=f"{col_name} vs {col2_name}",
                            xaxis_title=col_name,
                            yaxis_title=col2_name,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(47,47,47,0.05)',
                            font=dict(color='#2F2F2F'),
                            title_font=dict(size=16, color='#DAA520')
                        )
                        charts['scatter'] = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)
        
        except Exception as e:
            print(f"Error generating charts: {e}")
    
    return charts

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            df_dict, headers = read_csv_simple('data/processed_data.csv')
            numeric_cols = get_numeric_columns(df_dict)
            
            target_column = request.form.get('target_column')
            feature_columns = request.form.getlist('feature_columns')
            model_type = request.form.get('model_type', 'linear')
            
            if not target_column or not feature_columns:
                flash('Please select target and feature columns', 'warning')
                return redirect(request.url)
            
            # Prepare data for modeling
            X_list = []
            y_list = []
            
            # Build feature matrix and target vector
            for i in range(len(df_dict[target_column])):
                try:
                    # Get target value
                    y_val = df_dict[target_column][i]
                    if not isinstance(y_val, (int, float)):
                        continue
                        
                    # Get feature values
                    x_vals = []
                    valid_row = True
                    for feature_col in feature_columns:
                        if i < len(df_dict[feature_col]):
                            x_val = df_dict[feature_col][i]
                            if isinstance(x_val, (int, float)):
                                x_vals.append(x_val)
                            else:
                                valid_row = False
                                break
                        else:
                            valid_row = False
                            break
                    
                    if valid_row and len(x_vals) == len(feature_columns):
                        X_list.append(x_vals)
                        y_list.append(y_val)
                        
                except Exception:
                    continue
            
            if len(X_list) < 10:  # Need at least 10 data points
                flash('Not enough valid numeric data for prediction. Need at least 10 data points.', 'error')
                return redirect(request.url)
            
            # Convert to numpy arrays
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Split data
            test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adjust test size based on data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model
            if model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif model_type == 'forest':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = LinearRegression()  # default
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'mae': round(mae, 3),
                'mse': round(mse, 3),
                'rmse': round(np.sqrt(mse), 3),
                'r2': round(r2, 3),
                'target_column': target_column,
                'feature_columns': feature_columns,
                'model_type': model_type,
                'task_type': 'regression',
                'model_performance': 'Excellent' if r2 > 0.8 else 'Good' if r2 > 0.6 else 'Fair',
                'future_predictions': y_pred[:8].tolist() if len(y_pred) >= 8 else y_pred.tolist()
            }
            
            return render_template('predict.html', results=results, show_results=True)
            
        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'error')
            return redirect(request.url)
    
    # GET request
    try:
        df_dict, headers = read_csv_simple('data/processed_data.csv')
        numeric_cols = get_numeric_columns(df_dict)
        
        if not numeric_cols:
            flash('No numeric columns found for prediction. Please upload data with numeric values.', 'warning')
            return redirect(url_for('upload_file'))
            
        return render_template('predict.html', 
                             numeric_columns=numeric_cols, 
                             all_columns=headers, 
                             show_results=False)
    except:
        flash('Please upload data first', 'warning')
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)