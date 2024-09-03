import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting

from flask import Flask, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from flask_cors import CORS
from loguru import logger
cors = CORS()
app = Flask(__name__)
CORS(app)

def clean_numeric_data(df):
    """ Convert columns to numeric, if possible, and handle errors """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def generate_histogram(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name].dropna(), kde=True)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

def generate_boxplot(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column_name].dropna())
    plt.title(f'Boxplot of {column_name}')
    plt.xlabel(column_name)

def generate_scatter_plot(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[df.columns[0]].dropna(), y=df[df.columns[1]].dropna())
    plt.title(f'Scatter Plot of {df.columns[0]} vs {df.columns[1]}')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])

def generate_line_plot(df, column_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df[df.columns[0]].dropna(), df[column_name].dropna())
    plt.title(f'Line Plot of {column_name}')
    plt.xlabel(df.columns[0])
    plt.ylabel(column_name)

def generate_bar_plot(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df[df.columns[0]].dropna(), y=df[column_name].dropna())
    plt.title(f'Bar Plot of {column_name}')
    plt.xlabel(df[df.columns[0]].name)
    plt.ylabel(column_name)

def generate_pie_chart(df, column_name):
    plt.figure(figsize=(8, 8))
    df[column_name].dropna().value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'Pie Chart of {column_name}')

def generate_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Correlation Matrix')

def generate_pair_plot(df):
    plt.figure(figsize=(10, 10))
    sns.pairplot(df.dropna())
    plt.title('Pair Plot')

def generate_violin_plot(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[column_name].dropna())
    plt.title(f'Violin Plot of {column_name}')

def generate_joint_plot(df):
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=df[df.columns[0]].dropna(), y=df[df.columns[1]].dropna())
    plt.title('Joint Plot')

def generate_regression_plot(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[df.columns[0]].dropna(), y=df[df.columns[1]].dropna())
    plt.title('Regression Plot')

def generate_area_plot(df, column_name):
    plt.figure(figsize=(10, 6))
    plt.stackplot(df[df.columns[0]].dropna(), df[column_name].dropna())
    plt.title(f'Area Plot of {column_name}')

def generate_contour_plot(df):
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(df[df.columns[0]].dropna(), df[df.columns[1]].dropna())
    Z = df[df.columns[2]].dropna().values.reshape(X.shape)
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.title('Contour Plot')

def generate_3d_scatter_plot(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = df[df.columns[0]].dropna()
    y = df[df.columns[1]].dropna()
    z = df[df.columns[2]].dropna()

    if not (len(x) == len(y) == len(z)):
        raise ValueError("Mismatch in lengths of x, y, and z data. Ensure all columns are of the same length and non-empty.")

    ax.scatter(x, y, z)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    plt.title('3D Scatter Plot')

def generate_3d_surface_plot(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(df[df.columns[0]].dropna(), df[df.columns[1]].dropna())
    Z = df[df.columns[2]].dropna().values.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    plt.title('3D Surface Plot')

def generate_3d_contour_plot(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(df[df.columns[0]].dropna(), df[df.columns[1]].dropna())
    Z = df[df.columns[2]].dropna().values.reshape(X.shape)
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    plt.title('3D Contour Plot')

# Mapping plot types to functions
PLOT_TYPE_FUNCTIONS = {
    'histogram': generate_histogram,
    'boxplot': generate_boxplot,
    'scatter': generate_scatter_plot,
    'line': generate_line_plot,
    'bar': generate_bar_plot,
    'pie': generate_pie_chart,
    'heatmap': generate_heatmap,
    'pair': generate_pair_plot,
    'violin': generate_violin_plot,
    'joint': generate_joint_plot,
    'regression': generate_regression_plot,
    'area': generate_area_plot,
    'contour': generate_contour_plot,
    '3dscatter': generate_3d_scatter_plot,
    '3dsurface': generate_3d_surface_plot,
    '3dcontour': generate_3d_contour_plot,
}

def generate_plot(df, plot_type, column_name=None):
    """ Generate a plot based on the type and column name """
    df = clean_numeric_data(df)
    plt.close('all')
    plt.figure(figsize=(10, 6))
    
    if plot_type in PLOT_TYPE_FUNCTIONS:
        plot_function = PLOT_TYPE_FUNCTIONS[plot_type]
        try:
            if plot_type in ['scatter', 'heatmap', 'pair', '3dscatter', '3dsurface', '3dcontour']:
                plot_function(df)
            else:
                if column_name:
                    plot_function(df, column_name)
                else:
                    return None, "Column name required for this plot type"
        except Exception as e:
            return None, str(e)
    else:
        return None, "Unsupported plot type"
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img, None

@app.route('/', methods = ['GET'])
def home():
    return 'Home'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df = clean_numeric_data(df)
        
        summary = df.describe().to_json()
        
        return jsonify({"summary": summary}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/plot', methods=['POST'])
def generate_plot_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    plot_type = request.form.get('plot_type')
    column_name = request.form.get('column_name')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df = clean_numeric_data(df)
        
        if column_name and column_name not in df.columns:
            return jsonify({"error": "Invalid column name"}), 400
        
        img, error = generate_plot(df, plot_type, column_name)
        if error:
            logger.debug(error)
            return jsonify({"error": error}), 400
        
        return send_file(img, mimetype='image/png', as_attachment=True, download_name='plot.png')
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
