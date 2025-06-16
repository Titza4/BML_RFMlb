"""
Модуль для обработки ошибок и совместимости с различными операционными системами
"""
import os
import sys
import warnings
from functools import wraps

def windows_multiprocessing_fix():
    """
    Исправление проблем с multiprocessing на Windows
    """
    if os.name == 'nt':  # Windows
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            print("Windows multiprocessing fix применен успешно")
        except RuntimeError as e:
            if "context has already been set" in str(e):
                print("Multiprocessing context уже установлен")
            else:
                print(f"Предупреждение: {e}")
        except Exception as e:
            print(f"Ошибка при применении multiprocessing fix: {e}")

def safe_sklearn_operation(func):
    """
    Декоратор для безопасного выполнения операций sklearn с обработкой ошибок
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Устанавливаем n_jobs=1 для Windows совместимости
            if 'n_jobs' in kwargs and os.name == 'nt':
                kwargs['n_jobs'] = 1
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Ошибка в {func.__name__}: {str(e)}")
            # Возвращаем безопасное значение по умолчанию
            if 'predict' in func.__name__:
                return None
            return None
    return wrapper

def check_environment():
    """
    Проверка среды выполнения и вывод информации о системе
    """
    print("=== ИНФОРМАЦИЯ О СРЕДЕ ВЫПОЛНЕНИЯ ===")
    print(f"Операционная система: {os.name}")
    print(f"Платформа: {sys.platform}")
    print(f"Python версия: {sys.version}")
    
    # Проверка основных библиотек
    libraries = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'неизвестна')
            print(f"{lib}: {version}")
        except ImportError:
            print(f"{lib}: НЕ УСТАНОВЛЕНА")
    
    print("=" * 40)

def suppress_warnings():
    """
    Подавление предупреждений для чистого вывода
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Специфичные предупреждения для sklearn
    warnings.filterwarnings('ignore', message='.*joblib.*')
    warnings.filterwarnings('ignore', message='.*multiprocessing.*')

class SafeClusteringMixin:
    """
    Миксин для безопасной кластеризации с обработкой ошибок
    """
    
    def safe_kmeans(self, data, n_clusters=5, **kwargs):
        """Безопасный K-means"""
        try:
            from sklearn.cluster import KMeans
            kwargs['n_jobs'] = 1  # Принудительно для Windows
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            return kmeans.fit_predict(data)
        except Exception as e:
            print(f"Ошибка K-means: {e}")
            return np.zeros(len(data))
    
    def safe_hierarchical(self, data, n_clusters=5, **kwargs):
        """Безопасная иерархическая кластеризация"""
        try:
            from sklearn.cluster import AgglomerativeClustering
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            return hierarchical.fit_predict(data)
        except Exception as e:
            print(f"Ошибка Hierarchical: {e}")
            return np.zeros(len(data))
    
    def safe_gmm(self, data, n_components=5, **kwargs):
        """Безопасный GMM"""
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=n_components, random_state=42, **kwargs)
            return gmm.fit_predict(data)
        except Exception as e:
            print(f"Ошибка GMM: {e}")
            return np.zeros(len(data))

def create_fallback_data():
    """
    Создание демонстрационных данных для тестирования
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    # Генерация демонстрационных данных
    n_customers = 1000
    n_transactions = 5000
    
    customer_ids = np.random.randint(1000, 5000, n_transactions)
    
    # Создание реалистичных дат
    start_date = datetime(2010, 12, 1)
    end_date = datetime(2011, 12, 9)
    dates = [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) 
             for _ in range(n_transactions)]
    
    data = pd.DataFrame({
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_transactions)],
        'StockCode': [f'PROD{np.random.randint(1, 1000):03d}' for _ in range(n_transactions)],
        'Description': [f'Product {np.random.randint(1, 100)}' for _ in range(n_transactions)],
        'Quantity': np.random.randint(1, 20, n_transactions),
        'InvoiceDate': dates,
        'UnitPrice': np.random.uniform(1, 100, n_transactions).round(2),
        'CustomerID': customer_ids.astype(float),
        'Country': np.random.choice(['United Kingdom', 'France', 'Germany'], n_transactions)
    })
    
    # Добавляем немного пропусков для реалистичности
    missing_indices = np.random.choice(n_transactions, size=int(0.1 * n_transactions), replace=False)
    data.loc[missing_indices, 'CustomerID'] = np.nan
    
    return data

if __name__ == "__main__":
    # Тестирование модуля
    check_environment()
    windows_multiprocessing_fix()
    suppress_warnings()
    
    print("\nМодуль error_handler готов к использованию")
