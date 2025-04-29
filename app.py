import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload_page')
def upload_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        ndvi = calculate_ndvi(filepath)
        mean_ndvi, min_ndvi, max_ndvi, std_ndvi, healthy_percentage, problematic_percentage = calculate_ndvi_stats(ndvi)
        anomaly_map = detect_anomalies(ndvi)

        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Resultado NDVI")
        plt.imshow(ndvi, cmap='RdYlGn')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Anomalias Detectadas")
        plt.imshow(anomaly_map, cmap='gray')
        plt.colorbar()
        plt.savefig(os.path.join(static_dir, 'ndvi_anomalies.png'))
        plt.close()

        plt.figure()
        plt.hist(ndvi.ravel(), bins=50, color='green')
        plt.title('Histograma do NDVI')
        plt.xlabel('Valor NDVI')
        plt.ylabel('Frequência')
        plt.savefig(os.path.join(static_dir, 'ndvi_histogram.png'))
        plt.close()

        stats_file = os.path.join(static_dir, 'ndvi_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f'Média do NDVI: {mean_ndvi:.4f}\n')
            f.write(f'Valor Mínimo do NDVI: {min_ndvi:.4f}\n')
            f.write(f'Valor Máximo do NDVI: {max_ndvi:.4f}\n')
            f.write(f'Desvio Padrão do NDVI: {std_ndvi:.4f}\n')
            f.write(f'Percentual de Áreas Saudáveis (NDVI > 0.3): {healthy_percentage:.2f}%\n')
            f.write(f'Percentual de Áreas Problemáticas (NDVI < 0): {problematic_percentage:.2f}%\n')

        return redirect(url_for('result'))

    return redirect('/upload_page')

@app.route('/result')
def result():
    return render_template('result.html')

def calculate_ndvi(image_path):
    image = cv2.imread(image_path)
    red = image[:, :, 2].astype(float)
    nir = image[:, :, 1].astype(float)
    bottom = (nir + red)
    bottom[bottom == 0] = 1
    ndvi = (nir - red) / bottom
    return ndvi

def calculate_ndvi_stats(ndvi):
    mean_ndvi = np.mean(ndvi)
    min_ndvi = np.min(ndvi)
    max_ndvi = np.max(ndvi)
    std_ndvi = np.std(ndvi)
    healthy_percentage = np.sum(ndvi > 0.3) / ndvi.size * 100
    problematic_percentage = np.sum(ndvi < 0) / ndvi.size * 100
    return mean_ndvi, min_ndvi, max_ndvi, std_ndvi, healthy_percentage, problematic_percentage

def detect_anomalies(ndvi):
    pixels = ndvi.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(pixels)
    labels = kmeans.labels_.reshape(ndvi.shape)
    cluster_0_mean = np.mean(ndvi[labels == 0])
    cluster_1_mean = np.mean(ndvi[labels == 1])
    anomaly_label = 0 if cluster_0_mean < cluster_1_mean else 1
    anomaly_map = (labels == anomaly_label).astype(int)
    return anomaly_map

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
