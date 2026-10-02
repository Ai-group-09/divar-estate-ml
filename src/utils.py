import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import jdatetime
import re

from datetime import datetime
from shapely.geometry import Point
from scipy import stats




def load_dataset(path):
    data = pd.read_csv(path)
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
        df = data.copy()
    return df




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
    



def calculate_real_price(data, price_col, year_col):
 
    inflation_rates = {
        1400: 0.40,
        1401: 0.46,
        1402: 0.41,
        1403: 0.325
    }

    cpi = {1399: 100}
    cpi[1400] = cpi[1399] * (1 + inflation_rates[1400])
    cpi[1401] = cpi[1400] * (1 + inflation_rates[1401])
    cpi[1402] = cpi[1401] * (1 + inflation_rates[1402])
    cpi[1403] = cpi[1402] * (1 + inflation_rates[1403])

    # ضریب تعدیل نسبت به سال پایه ۱۳۹۹
    adjustment_factor = {year: cpi[year] / 100 for year in cpi}

    df = data.copy()

    if "cat2_slug" in df.columns:
        df = df[df["cat2_slug"].isin(["residential-sell", "commercial-sell"])]

    df = df[df[year_col].isin([1400, 1401, 1402, 1403])]
    df = df[df[price_col].notna() & (df[price_col] > 0)]

    # میانگین قیمت اسمی
    nominal_mean = df.groupby(year_col)[price_col].mean()

    # میانگین قیمت حقیقی
    real_mean = {}
    for year in [1400, 1401, 1402, 1403]:
        if year in nominal_mean.index:
            real_mean[year] = nominal_mean[year] / adjustment_factor[year]

    real_mean = pd.Series(real_mean)

    comparison = pd.DataFrame({
        "nominal_mean": nominal_mean,
        "real_mean": real_mean
    })

    return comparison.round(2)




def prepare_amenities_gdf(
    data, amenities, iran_map, residential_cat):
    
    df_res = data[data['cat2_slug'].isin(residential_cat)].copy()
    
    existing = [col for col in amenities if col in df_res.columns]
    
    for col in existing:
        df_res[col] = df_res[col].astype(str).str.lower().str.strip().map({'true': 1, 'false': 0})
    
    df_res = df_res.dropna(subset=['location_latitude', 'location_longitude'])
    
    lat_min, lat_max = 24.5, 40.5
    lon_min, lon_max = 43.5, 64.0
    
    mask_bbox = (df_res['location_latitude'].between(lat_min, lat_max) & df_res['location_longitude'].between(lon_min, lon_max))
    df_res = df_res[mask_bbox].copy()
    
    props_gdf = gpd.GeoDataFrame(df_res[['city_slug'] + existing], 
                                 geometry=gpd.points_from_xy(df_res['location_longitude'], df_res['location_latitude']),
                                 crs='EPSG:4326')
    
    if iran_map.crs != props_gdf.crs:
        iran_same_crs = iran_map.to_crs(props_gdf.crs)
    else:
        iran_same_crs = iran_map
    
    props_gdf = gpd.sjoin(
        props_gdf,
        iran_same_crs[['name', 'geometry']],
        how='left',
        predicate='within'
    ).rename(columns={'name': 'province'})
    
    props_gdf = props_gdf.drop(columns=['index_right'], errors='ignore')
    
    props_gdf = props_gdf.dropna(subset=['province']).copy()
    
    return props_gdf





def plot_amenities_scatter(
    props_gdf,
    iran_map,
    amenities,
    colors=None,
    group_title='',
    sample_size=20000,
    ncols=2,
):
    n = len(amenities)
    
    # پیش‌فرض‌ها
    titles= amenities
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                  '#9b59b6', '#16a085'][:n]
    
    # محاسبه چیدمان
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 7 * nrows)
    )
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (amenity, title, color) in enumerate(zip(amenities, titles, colors)):
        ax = axes[i]
        
        iran_map.plot(ax=ax, color='#f5f5f5', edgecolor='#888888', linewidth=0.5)
        
        has_amenity = props_gdf[props_gdf[amenity] == 1]
        n_with = len(has_amenity)
        
        if n_with > sample_size:
            has_amenity = has_amenity.sample(sample_size, random_state=42)
        
        has_amenity.plot(ax=ax, color=color, markersize=2, alpha=0.5)
        
        ax.set_title(
            f'{title}\n({n_with:,} estate)',
            fontsize=13, fontweight='bold'
        )
        ax.axis('off')
    
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    if group_title:
        fig.suptitle(group_title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    
    
    
def plot_amenities_choropleth(
    props_gdf,
    iran_map,
    amenities,
    group_title='',
    min_listings=500,
    ncols=2,
    figsize_per_plot=(7, 7),
):
    n = len(amenities)
    
    titles = amenities
    cmaps = ['YlOrRd', 'YlGnBu', 'Greens', 'Oranges', 'Purples', 'Reds'][:n]
    
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (amenity, title, cmap) in enumerate(zip(amenities, titles, cmaps)):
        ax = axes[i]
        
        stats = props_gdf.groupby('province').agg(
            ratio=(amenity, 'mean'),
            total=('city_slug', 'size')
        ).reset_index()
        stats = stats[stats['total'] >= min_listings]
        
        iran_stats = iran_map.merge(
            stats, left_on='name', right_on='province', how='left'
        )
        
        if not isinstance(iran_stats, gpd.GeoDataFrame):
            iran_stats = gpd.GeoDataFrame(
                iran_stats, geometry='geometry', crs=iran_map.crs
            )
        
        bg = iran_stats[iran_stats['ratio'].isna()]
        if not bg.empty:
            bg.plot(ax=ax, color='#f0f0f0',
                    edgecolor='#cccccc', linewidth=0.5)
        
        colored = iran_stats.dropna(subset=['ratio'])
        if not colored.empty:
            vmin_data = colored['ratio'].min()
            vmax_data = colored['ratio'].max()
            
            if vmax_data - vmin_data < 0.01:
                vmin_data = max(0, vmin_data - 0.01)
                vmax_data = min(1, vmax_data + 0.01)
            
            colored.plot(column='ratio', cmap=cmap, legend=True, ax=ax, edgecolor='black', 
                         linewidth=0.5, vmin=vmin_data, vmax=vmax_data, 
                         legend_kwds={'label': 'Ratio', 'shrink': 0.6, 'format': '%.2f'})
        else:
            if not iran_stats.empty:
                iran_stats.plot(
                    ax=ax, color='#f0f0f0',
                    edgecolor='#cccccc', linewidth=0.5
                )
            ax.text(0.5, 0.5, f'Insufficient data\n(min={min_listings})', ha='center', 
                    va='center', transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_title(f'{title}', fontsize=13, fontweight='bold')
        ax.axis('off')
    
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    if group_title:
        fig.suptitle(f'Ratio of properties by province — {group_title}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()



def test_business_deed_effect(data, deed_col="has_business_deed", price_col="price_value"):
    df = data.copy()

    # نرمال‌سازی سند
    df[deed_col] = (
        df[deed_col]
        .replace({"true": True, "false": False, "unselect": pd.NA})
        .astype("boolean")
    )

    df = df.dropna(subset=[deed_col, price_col])
    df = df[df[price_col] > 0]

    # حذف پرت‌ها
    q1 = df[price_col].quantile(0.25)
    q3 = df[price_col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[price_col] >= q1 - 1.5 * iqr) &
            (df[price_col] <= q3 + 1.5 * iqr)]

    with_deed = df[df[deed_col] == True][price_col]
    without_deed = df[df[deed_col] == False][price_col]

    if len(with_deed) < 30 or len(without_deed) < 30:
        print("داده کافی برای مقایسه وجود ندارد.")
        return None

    # آزمون آماری
    _, p_value = stats.mannwhitneyu(with_deed, without_deed, alternative="two-sided")

    # اندازه اثر
    pooled_std = np.sqrt((with_deed.std()**2 + without_deed.std()**2) / 2)
    cohens_d = (with_deed.mean() - without_deed.mean()) / pooled_std if pooled_std > 0 else 0

    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"

    result = pd.DataFrame([{
        "group": "با سند",
        "count": len(with_deed),
        "mean": round(with_deed.mean()),
        "median": round(with_deed.median())
    }, {
        "group": "بدون سند",
        "count": len(without_deed),
        "mean": round(without_deed.mean()),
        "median": round(without_deed.median())
    }])

    print("=== نتیجه آزمون سند تجاری ===")
    print(result.to_string(index=False))
    print(f"\np-value          : {p_value:.6e}")
    print(f"significant      : {p_value < 0.05}")
    print(f"Cohen's d        : {cohens_d:.4f}")
    print(f"effect size      : {effect}")

    return {
        "result_table": result,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": round(cohens_d, 4),
        "effect_size": effect,
        "with_deed": with_deed,
        "without_deed": without_deed
    }





def test_amenity_effect_on_price(data, amenities, price_col='price_value'):
    if isinstance(amenities, str):
        amenities = [amenities]

    results = []

    for amenity in amenities:
        df = data.copy()

        df[amenity] = (df[amenity].replace({"true": True, "false": False, "unselect": pd.NA}).astype("boolean")
)

        df = df.dropna(subset=[amenity, price_col])
        df = df[df[price_col] > 0]

        # حذف پرت‌ها
        q1 = df[price_col].quantile(0.25)
        q3 = df[price_col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[price_col] >= q1 - 1.5 * iqr) & 
                (df[price_col] <= q3 + 1.5 * iqr)]

        has = df[df[amenity] == True][price_col]
        not_has = df[df[amenity] == False][price_col]

        if len(has) < 30 or len(not_has) < 30:
            continue

        # آزمون
        _, p_value = stats.mannwhitneyu(has, not_has, alternative="two-sided")

        # اندازه اثر
        pooled_std = np.sqrt((has.std()**2 + not_has.std()**2) / 2)
        cohens_d = (has.mean() - not_has.mean()) / pooled_std if pooled_std > 0 else 0

        # تفسیر اندازه اثر
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect = "negligible"
        elif abs_d < 0.5:
            effect = "small"
        elif abs_d < 0.8:
            effect = "medium"
        else:
            effect = "large"

        results.append({
            "amenity": amenity,
            "n_with": len(has),
            "n_without": len(not_has),
            "mean_with": round(has.mean()),
            "mean_without": round(not_has.mean()),
            "median_with": round(has.median()),
            "median_without": round(not_has.median()),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": round(cohens_d, 4),
            "effect_size": effect
        })

    return pd.DataFrame(results)


def persian_to_latin(text):
    """تبدیل اعداد و کاراکترهای فارسی به لاتین"""
    if pd.isna(text):
        return text
    text = str(text)
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    latin_digits = '0123456789'
    
    trans_table = str.maketrans(
        persian_digits + arabic_digits,
        latin_digits + latin_digits
    )
    return text.translate(trans_table)



def extract_year(val):
    """استخراج سال ساخت به صورت عددی"""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    
    if 'قبل از' in val or 'قبل' in val:
        return 1369  
    
    match = re.search(r'1[34]\d{2}', val)
    if match:
        year = int(match.group())

        return year




def normalize_bool(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if val in ['true', '1', 'yes', 'بله', 'دارد']:
        return True
    if val in ['false', '0', 'no', 'خیر', 'ندارد']:
        return False
    if val in ['unselect', '', 'nan', 'none']:
        return np.nan
    return np.nan





def to_jalali(date):
    """تبدیل تاریخ میلادی به شمسی"""
    if pd.isna(date):
        return np.nan
    try:
        g = pd.to_datetime(date)
        j = jdatetime.date.fromgregorian(date=g)
        return j
    except:
        return np.nan




def parse_number(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    # حذف کاما و فاصله
    val = val.replace(',', '').replace('،', '').strip()
    try:
        return float(val)
    except:
        # اگر داخل متن عدد بود، استخراج کن
        match = re.search(r'\d+(\.\d+)?', val)
        if match:
            return float(match.group())
        return np.nan




def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace('ي', 'ی').replace('ك', 'ک') 
    text = re.sub(r'\u200c', ' ', text)                 
    text = re.sub(r'\s+', ' ', text)                   
    return text.strip()