import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RFMAnalyzer:
    """
    Класс для расчета RFM показателей
    """
    
    def __init__(self, analysis_date=None):
        """
        Args:
            analysis_date (datetime): Дата анализа (по умолчанию - максимальная дата в данных + 1 день)
        """
        self.analysis_date = analysis_date
    
    def calculate_rfm(self, data):
        """
        Расчет RFM показателей для каждого клиента
        
        Args:
            data (pd.DataFrame): Подготовленные данные транзакций
            
        Returns:
            pd.DataFrame: RFM данные по клиентам
        """
        # Определение даты анализа
        if self.analysis_date is None:
            self.analysis_date = data['InvoiceDate'].max() + timedelta(days=1)
        
        # Группировка по клиентам и расчет показателей
        rfm = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.analysis_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        }).reset_index()
        
        # Переименование столбцов
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Обработка отрицательных Monetary (клиенты с чистым возвратом)
        rfm = rfm[rfm['Monetary'] > 0]  # Исключаем клиентов только с возвратами
        
        # Добавление RFM скоров
        rfm = self._calculate_rfm_scores(rfm)
        
        return rfm
    
    def _calculate_rfm_scores(self, rfm):
        """
        Расчет RFM скоров (квантильное разбиение на 5 групп)
        """
        # Recency: меньше = лучше (1 - недавно, 5 - давно)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        
        # Frequency: больше = лучше (5 - часто, 1 - редко)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Monetary: больше = лучше (5 - много тратит, 1 - мало тратит)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Общий RFM скор
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm
    
    def segment_customers_by_rfm(self, rfm):
        """
        Сегментация клиентов на основе RFM скоров
        """
        def rfm_level(df):
            if df['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif df['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif df['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif df['RFM_Score'] in ['453', '513', '533', '532', '521', '522', '523', '551', '552']:
                return 'New Customers'
            elif df['RFM_Score'] in ['534', '343', '334', '343', '244', '334']:
                return 'Promising'
            elif df['RFM_Score'] in ['331', '321', '231', '241', '251']:
                return 'Need Attention'
            elif df['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'About to Sleep'
            elif df['RFM_Score'] in ['143', '142', '135', '125', '124']:
                return 'At Risk'
            elif df['RFM_Score'] in ['111', '112', '121', '131', '141', '151']:
                return 'Cannot Lose Them'
            elif df['RFM_Score'] in ['155', '154', '144', '214', '215', '115']:
                return 'Hibernating'
            else:
                return 'Lost'
        
        rfm['Segment'] = rfm.apply(rfm_level, axis=1)
        return rfm
    
    def get_segment_summary(self, rfm):
        """
        Получение сводки по сегментам
        """
        segment_summary = rfm.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        }).round(2)
        
        # Flatten column names
        segment_summary.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue']
        
        # Add percentage
        segment_summary['Percentage'] = (segment_summary['Count'] / segment_summary['Count'].sum() * 100).round(2)
        
        return segment_summary.sort_values('Total_Revenue', ascending=False)
