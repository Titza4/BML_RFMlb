import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import os
import warnings

# Исправление для Windows multiprocessing
if os.name == 'nt':  # Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

class ClusteringAnalyzer:
    """
    Класс для анализа кластеризации RFM данных
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
    
    def prepare_for_clustering(self, rfm_data):
        """
        Подготовка RFM данных для кластеризации
        
        Args:
            rfm_data (pd.DataFrame): RFM данные
            
        Returns:
            tuple: (scaled_data, scaler)
        """
        # Выбираем только RFM признаки
        features = ['Recency', 'Frequency', 'Monetary']
        rfm_features = rfm_data[features].copy()
        
        # Логарифмирование для нормализации распределений
        rfm_features['Recency_log'] = np.log1p(rfm_features['Recency'])
        rfm_features['Frequency_log'] = np.log1p(rfm_features['Frequency'])
        rfm_features['Monetary_log'] = np.log1p(rfm_features['Monetary'])
        
        # Использование логарифмированных признаков
        log_features = ['Recency_log', 'Frequency_log', 'Monetary_log']
        
        # Стандартизация
        scaled_data = self.scaler.fit_transform(rfm_features[log_features])
        
        return scaled_data, self.scaler
    
    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Поиск оптимального количества кластеров методом локтя и силуэта
        """
        inertias = []
        silhouette_scores = []
        
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            try:
                # Используем n_jobs=1 для избежания multiprocessing проблем на Windows
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, n_jobs=1)
                labels = kmeans.fit_predict(data)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(data, labels))
            except Exception as e:
                print(f"Ошибка при обработке k={k}: {str(e)}")
                inertias.append(np.nan)
                silhouette_scores.append(np.nan)
        
        # Построение графиков
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Метод локтя
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Количество кластеров')
        ax1.set_ylabel('Инерция')
        ax1.set_title('Метод локтя')
        ax1.grid(True)
        
        # Silhouette score
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Количество кластеров')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Рекомендация (исключаем NaN значения)
        valid_scores = [(i, score) for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
        if valid_scores:
            best_idx = max(valid_scores, key=lambda x: x[1])[0]
            best_k = K_range[best_idx]
            print(f"Рекомендуемое количество кластеров (по Silhouette): {best_k}")
        else:
            best_k = 5  # Значение по умолчанию
            print("Не удалось определить оптимальное количество кластеров, используется 5")
        
        return best_k
    
    def apply_clustering_methods(self, data, n_clusters=5):
        """
        Применение различных методов кластеризации
        
        Args:
            data (np.array): Подготовленные данные
            n_clusters (int): Количество кластеров
            
        Returns:
            dict: Результаты кластеризации для каждого метода
        """
        results = {}
        
        # K-Means с исправлением для Windows
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, n_jobs=1)
            results['K-means'] = kmeans.fit_predict(data)
        except Exception as e:
            print(f"Ошибка K-means: {str(e)}")
            results['K-means'] = np.zeros(len(data))
        
        # Hierarchical Clustering
        try:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            results['Hierarchical'] = hierarchical.fit_predict(data)
        except Exception as e:
            print(f"Ошибка Hierarchical: {str(e)}")
            results['Hierarchical'] = np.zeros(len(data))
        
        # Gaussian Mixture Model
        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            results['GMM'] = gmm.fit_predict(data)
        except Exception as e:
            print(f"Ошибка GMM: {str(e)}")
            results['GMM'] = np.zeros(len(data))
        
        # Spectral Clustering с исправлением для Windows
        try:
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, n_jobs=1)
            results['Spectral'] = spectral.fit_predict(data)
        except Exception as e:
            print(f"Ошибка Spectral: {str(e)}")
            results['Spectral'] = np.zeros(len(data))
        
        # DBSCAN (автоматическое определение количества кластеров)
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=1)
            results['DBSCAN'] = dbscan.fit_predict(data)
        except Exception as e:
            print(f"Ошибка DBSCAN: {str(e)}")
            results['DBSCAN'] = np.zeros(len(data))
        
        return results
    
    def evaluate_clustering(self, data, clustering_results):
        """
        Оценка качества кластеризации
        
        Args:
            data (np.array): Данные
            clustering_results (dict): Результаты кластеризации
            
        Returns:
            dict: Метрики качества для каждого метода
        """
        evaluation = {}
        
        for method, labels in clustering_results.items():
            # Пропускаем если все точки в одном кластере или есть шум
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                evaluation[method] = {
                    'silhouette': np.nan,
                    'calinski_harabasz': np.nan,
                    'davies_bouldin': np.nan,
                    'n_clusters': len(unique_labels)
                }
                continue
            
            try:
                # Для DBSCAN исключаем шумовые точки (-1) из расчета метрик
                if method == 'DBSCAN' and -1 in unique_labels:
                    valid_mask = labels != -1
                    if np.sum(valid_mask) < 2 or len(np.unique(labels[valid_mask])) < 2:
                        evaluation[method] = {
                            'silhouette': np.nan,
                            'calinski_harabasz': np.nan,
                            'davies_bouldin': np.nan,
                            'n_clusters': len(unique_labels) - 1  # исключаем -1
                        }
                        continue
                    
                    data_valid = data[valid_mask]
                    labels_valid = labels[valid_mask]
                    
                    silhouette = silhouette_score(data_valid, labels_valid)
                    calinski_harabasz = calinski_harabasz_score(data_valid, labels_valid)
                    davies_bouldin = davies_bouldin_score(data_valid, labels_valid)
                    n_clusters_effective = len(np.unique(labels_valid))
                else:
                    silhouette = silhouette_score(data, labels)
                    calinski_harabasz = calinski_harabasz_score(data, labels)
                    davies_bouldin = davies_bouldin_score(data, labels)
                    n_clusters_effective = len(unique_labels)
                
                evaluation[method] = {
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'davies_bouldin': davies_bouldin,
                    'n_clusters': n_clusters_effective
                }
            except Exception as e:
                print(f"Ошибка при оценке {method}: {str(e)}")
                evaluation[method] = {
                    'silhouette': np.nan,
                    'calinski_harabasz': np.nan,
                    'davies_bouldin': np.nan,
                    'n_clusters': len(unique_labels)
                }
        
        return evaluation
    
    def plot_clustering_comparison(self, data, clustering_results):
        """
        Визуализация результатов кластеризации
        """
        # PCA для визуализации
        try:
            pca_data = self.pca.fit_transform(data)
        except Exception as e:
            print(f"Ошибка PCA: {str(e)}")
            # Используем первые две координаты если PCA не работает
            pca_data = data[:, :2] if data.shape[1] >= 2 else np.column_stack([data[:, 0], data[:, 0]])
        
        n_methods = len(clustering_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (method, labels) in enumerate(clustering_results.items()):
            if i < len(axes):
                try:
                    scatter = axes[i].scatter(pca_data[:, 0], pca_data[:, 1], 
                                            c=labels, cmap='tab10', alpha=0.7)
                    axes[i].set_title(f'{method}\nКластеров: {len(np.unique(labels))}')
                    axes[i].set_xlabel('PC1' if pca_data.shape[1] > 1 else 'Feature 1')
                    axes[i].set_ylabel('PC2' if pca_data.shape[1] > 1 else 'Feature 1')
                    plt.colorbar(scatter, ax=axes[i])
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Ошибка отображения\n{method}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{method} - Ошибка')
        
        # Убираем лишние подграфики
        for i in range(len(clustering_results), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.show()
