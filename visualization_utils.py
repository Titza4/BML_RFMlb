import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class VisualizationUtils:
    """
    Класс для визуализации результатов RFM анализа
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_rfm_distributions(self, rfm_data):
        """
        Построение распределений RFM показателей
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Исходные распределения
        axes[0, 0].hist(rfm_data['Recency'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Распределение Recency')
        axes[0, 0].set_xlabel('Дни с последней покупки')
        
        axes[0, 1].hist(rfm_data['Frequency'], bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Распределение Frequency')
        axes[0, 1].set_xlabel('Количество покупок')
        
        axes[0, 2].hist(rfm_data['Monetary'], bins=50, alpha=0.7, color='salmon')
        axes[0, 2].set_title('Распределение Monetary')
        axes[0, 2].set_xlabel('Сумма покупок')
        
        # Логарифмированные распределения
        axes[1, 0].hist(np.log1p(rfm_data['Recency']), bins=50, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Log(Recency + 1)')
        axes[1, 0].set_xlabel('Log(Дни + 1)')
        
        axes[1, 1].hist(np.log1p(rfm_data['Frequency']), bins=50, alpha=0.7, color='lightgreen')
        axes[1, 1].set_title('Log(Frequency + 1)')
        axes[1, 1].set_xlabel('Log(Покупки + 1)')
        
        axes[1, 2].hist(np.log1p(rfm_data['Monetary']), bins=50, alpha=0.7, color='salmon')
        axes[1, 2].set_title('Log(Monetary + 1)')
        axes[1, 2].set_xlabel('Log(Сумма + 1)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_clusters_3d(self, rfm_data):
        """
        3D визуализация кластеров
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Логарифмирование для лучшей визуализации
        x = np.log1p(rfm_data['Recency'])
        y = np.log1p(rfm_data['Frequency'])
        z = np.log1p(rfm_data['Monetary'])
        
        scatter = ax.scatter(x, y, z, c=rfm_data['Cluster'], 
                           cmap='tab10', alpha=0.6, s=50)
        
        ax.set_xlabel('Log(Recency + 1)')
        ax.set_ylabel('Log(Frequency + 1)')
        ax.set_zlabel('Log(Monetary + 1)')
        ax.set_title('3D визуализация RFM кластеров')
        
        plt.colorbar(scatter)
        plt.show()
    
    def plot_cluster_characteristics(self, rfm_data):
        """
        Характеристики кластеров
        """
        cluster_summary = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Recency по кластерам
        cluster_summary['Recency'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Средняя Recency по кластерам')
        axes[0].set_ylabel('Дни')
        axes[0].tick_params(axis='x', rotation=0)
        
        # Frequency по кластерам
        cluster_summary['Frequency'].plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Средняя Frequency по кластерам')
        axes[1].set_ylabel('Покупки')
        axes[1].tick_params(axis='x', rotation=0)
        
        # Monetary по кластерам
        cluster_summary['Monetary'].plot(kind='bar', ax=axes[2], color='salmon')
        axes[2].set_title('Средняя Monetary по кластерам')
        axes[2].set_ylabel('Сумма')
        axes[2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_heatmap(self, rfm_data):
        """
        Тепловая карта характеристик кластеров
        """
        # Нормализация данных для лучшей визуализации
        cluster_summary = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        # Нормализация по столбцам (0-1)
        cluster_summary_norm = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_summary_norm.T, annot=True, cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Нормализованное значение'})
        plt.title('Тепловая карта характеристик кластеров')
        plt.ylabel('RFM показатели')
        plt.xlabel('Кластер')
        plt.show()
    
    def plot_cluster_sizes(self, rfm_data):
        """
        Размеры кластеров
        """
        cluster_sizes = rfm_data['Cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cluster_sizes.index, cluster_sizes.values, 
                      color=sns.color_palette("husl", len(cluster_sizes)))
        
        # Добавление значений на столбцы
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Размеры кластеров')
        plt.xlabel('Кластер')
        plt.ylabel('Количество клиентов')
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    
    def plot_customer_journey(self, rfm_data):
        """
        Анализ пути клиента (Recency vs Frequency vs Monetary)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Recency vs Frequency
        scatter1 = axes[0].scatter(rfm_data['Recency'], rfm_data['Frequency'], 
                                  c=rfm_data['Cluster'], cmap='tab10', alpha=0.6)
        axes[0].set_xlabel('Recency (дни)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Recency vs Frequency')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Frequency vs Monetary
        scatter2 = axes[1].scatter(rfm_data['Frequency'], rfm_data['Monetary'], 
                                  c=rfm_data['Cluster'], cmap='tab10', alpha=0.6)
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Monetary')
        axes[1].set_title('Frequency vs Monetary')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Recency vs Monetary
        scatter3 = axes[2].scatter(rfm_data['Recency'], rfm_data['Monetary'], 
                                  c=rfm_data['Cluster'], cmap='tab10', alpha=0.6)
        axes[2].set_xlabel('Recency (дни)')
        axes[2].set_ylabel('Monetary')
        axes[2].set_title('Recency vs Monetary')
        plt.colorbar(scatter3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
