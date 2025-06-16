"""
Конфигурационный файл для RFM анализа
"""

# Параметры данных
DATA_CONFIG = {
    'csv_file': 'customer_segmentation_project.csv',
    'encoding': 'ISO-8859-1',
    'date_column': 'InvoiceDate',
    'customer_id_column': 'CustomerID',
    'required_columns': ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
}

# Параметры RFM анализа
RFM_CONFIG = {
    'rfm_features': ['Recency', 'Frequency', 'Monetary'],
    'score_bins': 5,  # Количество квантилей для RFM скоров
    'min_monetary': 0,  # Минимальная сумма для включения клиента
    'use_log_transform': True  # Использовать логарифмирование
}

# Параметры кластеризации
CLUSTERING_CONFIG = {
    'n_clusters_range': (2, 10),
    'default_n_clusters': 5,
    'random_state': 42,
    'algorithms': {
        'kmeans': {'n_init': 10},
        'hierarchical': {'linkage': 'ward'},
        'gmm': {'covariance_type': 'full'},
        'spectral': {'affinity': 'rbf'},
        'dbscan': {'eps': 0.5, 'min_samples': 5}
    }
}

# Параметры визуализации
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),
    'style': 'seaborn-v0_8',
    'palette': 'husl',
    'dpi': 100,
    'alpha': 0.7
}

# Пороги для обработки выбросов
OUTLIER_CONFIG = {
    'quantity_percentile': 0.99,
    'price_percentile': 0.99,
    'z_score_threshold': 3
}

# Бизнес-правила для сегментации
BUSINESS_RULES = {
    'champions': {
        'recency_max': 50,
        'frequency_min': 10,
        'monetary_min': 1000
    },
    'loyal': {
        'recency_max': 100,
        'frequency_min': 5
    },
    'lost': {
        'recency_min': 200
    },
    'new': {
        'frequency': 1
    }
}
