import pandas as pd
from datetime import datetime
import jdatetime


def load_dataset(path):
    data = pd.read_csv(path)
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    return data


def extract_shamsi_year(date_str):
    """
    تاریخ میلادی رو می‌گیره و سال شمسی رو برمی‌گردونه
    """
    try:
        if pd.isna(date_str):
            return None
        
        dt = datetime.strptime(str(date_str).strip(), "%Y-%m-%d %H:%M:%S")
        j_date = jdatetime.date.fromgregorian(date=dt.date())
        return j_date.year
    
    except Exception:
        return None
    

def calculate_real_price(data, price_value_col, shamsi_year_col):
    
    # فقط سال‌های ۱۴۰۰ تا ۱۴۰۳ رو نگه می‌داریم
    df_filtered = data[data[shamsi_year_col].isin([1400, 1401, 1402, 1403])].copy()
    df_filtered = df_filtered[df_filtered[price_value_col].notna()]
    
    # میانگین قیمت اسمی
    nominal_mean = df_filtered.groupby(shamsi_year_col)[price_value_col].mean()
    
    # نرخ تورم تقریبی (مرکز آمار)
    inflation_rates = {
        1400: 0.40,
        1401: 0.46,
        1402: 0.41,
        1403: 0.325   
    }
    
    # محاسبه ضریب تجمعی تورم (CPI نسبی)
    cpi = {1400: 100}  # سال پایه

    cpi[1401] = cpi[1400] * (1 + inflation_rates[1401])
    cpi[1402] = cpi[1401] * (1 + inflation_rates[1402])
    cpi[1403] = cpi[1402] * (1 + inflation_rates[1403])
    
    # ساخت دیکشنری ضریب تعدیل (نسبت به سال پایه)
    adjustment_factor = {year: cpi[year] / 100 for year in cpi}

    # محاسبه میانگین قیمت حقیقی
    real_mean = {}
    for year in [1400, 1401, 1402, 1403]:
        if year in nominal_mean.index:
            real_mean[year] = nominal_mean[year] / adjustment_factor[year]
    real_mean = pd.Series(real_mean)
    
    comparison = pd.DataFrame({
    'nominal_mean': nominal_mean,
    'real_mean': real_mean
    })
    
    return comparison