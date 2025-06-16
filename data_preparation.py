import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreparator:
    """
    Класс для подготовки данных транзакций к RFM-анализу
    """
    
    def __init__(self):
        self.outlier_threshold = 3  # Порог для определения выбросов (Z-score)
    
    def prepare_data(self, data):
        """
        Основной метод подготовки данных
        
        Args:
            data (pd.DataFrame): Исходные данные транзакций
            
        Returns:
            pd.DataFrame: Подготовленные данные
        """
        df = data.copy()
        
        # Преобразование типов данных
        df = self._convert_data_types(df)
        
        # Обработка выбросов
        df = self._handle_outliers(df)
        
        # Финальная валидация данных
        df = self._validate_data(df)
        
        return df
    
    def _convert_data_types(self, df):
        """
        Преобразование типов данных
        """
        # Преобразование даты
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Преобразование CustomerID в int (убираем .0)
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        # Убеждаемся, что числовые поля корректны
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        
        return df
    
    def _handle_outliers(self, df):
        """
        Обработка выбросов в данных
        """
        # Удаление крайних выбросов по Quantity (сохраняем возвраты)
        q99_quantity = df[df['Quantity'] > 0]['Quantity'].quantile(0.99)
        df = df[df['Quantity'] <= q99_quantity]
        
        # Удаление крайних выбросов по UnitPrice
        q99_price = df['UnitPrice'].quantile(0.99)
        df = df[df['UnitPrice'] <= q99_price]
        
        return df
    
    def _validate_data(self, df):
        """
        Финальная валидация и очистка данных
        """
        # Удаление записей с некорректными данными
        initial_count = len(df)
        
        # Убираем строки с NaN после преобразований
        df = df.dropna(subset=['Quantity', 'UnitPrice', 'CustomerID'])
        
        # Убираем транзакции с некорректными датами
        df = df[df['InvoiceDate'].notna()]
        
        print(f"Удалено {initial_count - len(df)} некорректных записей при валидации")
        
        return df
    
    def get_data_summary(self, df):
        """
        Получение сводки по подготовленным данным
        """
        summary = {
            'total_transactions': len(df),
            'unique_customers': df['CustomerID'].nunique(),
            'unique_products': df['StockCode'].nunique(),
            'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
            'total_revenue': df['TotalPrice'].sum(),
            'returns_count': (df['TotalPrice'] < 0).sum(),
            'returns_value': df[df['TotalPrice'] < 0]['TotalPrice'].sum()
        }
        
        return summary
