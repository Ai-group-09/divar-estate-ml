import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import jdatetime

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




def prepare_amenities_gdf(
    data, amenities, iran_map, residential_cat):
    
    # ۱. فیلتر املاک مسکونی
    df_res = data[data['cat2_slug'].isin(residential_cat)].copy()
    
    # ۲. چک ستون‌ها
    existing = [col for col in amenities if col in df_res.columns]
    
    # ۳. تبدیل امکانات به 0/1
    for col in existing:
        df_res[col] = df_res[col].astype(str).str.lower().str.strip().map({'true': 1, 'false': 0})
    
    # ۴. حذف NaN مختصات
    df_res = df_res.dropna(subset=['location_latitude', 'location_longitude'])
    
    # ۵. فیلتر Bounding Box
    lat_min, lat_max = 24.5, 40.5
    lon_min, lon_max = 43.5, 64.0
    
    mask_bbox = (df_res['location_latitude'].between(lat_min, lat_max) & df_res['location_longitude'].between(lon_min, lon_max))
    df_res = df_res[mask_bbox].copy()
    
    # ۶. ساخت GeoDataFrame
    props_gdf = gpd.GeoDataFrame(df_res[['city_slug'] + existing], 
                                 geometry=gpd.points_from_xy(df_res['location_longitude'], df_res['location_latitude']),
                                 crs='EPSG:4326')
    
    # ۷. آماده‌سازی نقشه با Buffer ۵ کیلومتری
    if iran_map.crs != props_gdf.crs:
        iran_same_crs = iran_map.to_crs(props_gdf.crs)
    else:
        iran_same_crs = iran_map
    
    # ۸. Spatial Join
    props_gdf = gpd.sjoin(
        props_gdf,
        iran_same_crs[['name', 'geometry']],
        how='left',
        predicate='within'
    ).rename(columns={'name': 'province'})
    
    props_gdf = props_gdf.drop(columns=['index_right'], errors='ignore')
    
    # ۹. حذف نقاط باقی‌مانده خارج از مرز
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
    figsize_per_plot=(7, 7),
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
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (amenity, title, color) in enumerate(zip(amenities, titles, colors)):
        ax = axes[i]
        
        # نقشه زمینه
        iran_map.plot(ax=ax, color='#f5f5f5', edgecolor='#888888', linewidth=0.5)
        
        # نقاط دارای امکانات
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
    
    # خاموش کردن پلات‌های خالی
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
    # High-contrast colormaps
    cmaps = ['YlOrRd', 'YlGnBu', 'Greens', 'Oranges', 'Purples', 'Reds'][:n]
    
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (amenity, title, cmap) in enumerate(zip(amenities, titles, cmaps)):
        ax = axes[i]
        
        # Province-level stats
        stats = props_gdf.groupby('province').agg(
            ratio=(amenity, 'mean'),
            total=('city_slug', 'size')
        ).reset_index()
        stats = stats[stats['total'] >= min_listings]
        
        # Merge with map
        iran_stats = iran_map.merge(
            stats, left_on='name', right_on='province', how='left'
        )
        
        if not isinstance(iran_stats, gpd.GeoDataFrame):
            iran_stats = gpd.GeoDataFrame(
                iran_stats, geometry='geometry', crs=iran_map.crs
            )
        
        # Background (provinces without data)
        bg = iran_stats[iran_stats['ratio'].isna()]
        if not bg.empty:
            bg.plot(ax=ax, color='#f0f0f0',
                    edgecolor='#cccccc', linewidth=0.5)
        
        # Colored map (provinces with data)
        colored = iran_stats.dropna(subset=['ratio'])
        if not colored.empty:
            # ✅ Dynamic range for higher contrast
            vmin_data = colored['ratio'].min()
            vmax_data = colored['ratio'].max()
            
            # Add small padding to avoid zero range
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




import pandas as pd
import numpy as np
from scipy import stats


def test_business_deed_effect(
    data,
    commercial_cat='commercial-sell',
    target_col='price_value',
    group_col='has_business_deed',
):
    """
    بررسی تأثیر سند تجاری بر قیمت فروش ملک تجاری.
    
    Returns: dict
        شامل میانگین‌ها، p-value، آزمون استفاده‌شده، اندازه اثر
    """
    # ۱. فیلتر املاک تجاری فروش
    df = data[data['cat2_slug'] == commercial_cat].copy()
    
    # ۲. تبدیل گروه به ۰/۱
    df[group_col] = (
        df[group_col]
        .astype(str).str.lower().str.strip()
        .map({'true': 1, 'false': 0})
    )
    
    # ۳. حذف NaN
    df = df.dropna(subset=[group_col, target_col])
    df[group_col] = df[group_col].astype(int)
    
    # ۴. جدا کردن دو گروه
    group_with = df[df[group_col] == 1][target_col]
    group_without = df[df[group_col] == 0][target_col]
    
    # ۵. آمار توصیفی
    desc = {
        'with_deed': {
            'count': len(group_with),
            'mean': group_with.mean(),
            'median': group_with.median(),
            'std': group_with.std(),
        },
        'without_deed': {
            'count': len(group_without),
            'mean': group_without.mean(),
            'median': group_without.median(),
            'std': group_without.std(),
        },
    }
    
    # ۶. آزمون نرمال بودن (روی نمونه)
    sample_with = group_with.sample(min(5000, len(group_with)), random_state=42)
    sample_without = group_without.sample(min(5000, len(group_without)), random_state=42)
    _, p_norm_with = stats.shapiro(sample_with)
    _, p_norm_without = stats.shapiro(sample_without)
    is_normal = (p_norm_with >= 0.05) and (p_norm_without >= 0.05)
    
    # ۷. آزمون فرض
    if is_normal:
        stat, p_value = stats.ttest_ind(group_with, group_without, equal_var=False)
        test_used = 'Welch t-test'
    else:
        stat, p_value = stats.mannwhitneyu(
            group_with, group_without, alternative='two-sided'
        )
        test_used = 'Mann-Whitney U'
    
    # ۸. اندازه اثر (Cohen's d)
    nx, ny = len(group_with), len(group_without)
    pooled_std = np.sqrt(
        ((nx - 1) * group_with.std()**2 + (ny - 1) * group_without.std()**2)
        / (nx + ny - 2)
    )
    cohens_d = ((group_with.mean() - group_without.mean()) / pooled_std if pooled_std > 0 else 0)
    
    # ۹. تفسیر اندازه اثر
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_label = 'negligible'
    elif abs_d < 0.5:
        effect_label = 'small'
    elif abs_d < 0.8:
        effect_label = 'medium'
    else:
        effect_label = 'large'
    
    # ۱۰. نتیجه‌گیری
    is_significant = p_value < 0.05
    
    return {
        'descriptive': desc,
        'normality': {
            'p_value_with': p_norm_with,
            'p_value_without': p_norm_without,
            'is_normal': is_normal,
        },
        'test': {
            'name': test_used,
            'statistic': stat,
            'p_value': p_value,
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'label': effect_label,
        },
        'conclusion': {
            'alpha': 0.05,
            'is_significant': is_significant,
            'message': (
                'Business deed has a significant effect on price.'
                if is_significant else
                'No significant effect observed.'
            ),
        },
    }




def test_amenity_effect_on_price(
    data,
    amenities,
    category='residential-sell',
    target_col='price_value',
):
    """
    آزمون تأثیر هر امکانات بر قیمت فروش.
        
    Returns
    -------
    pd.DataFrame
        نتایج آزمون برای هر امکانات
    """
    df = data[data['cat2_slug'] == category].copy()
    results = []
    
    for amenity in amenities:
        if amenity not in df.columns:
            continue
        
        # تبدیل به ۰/۱/NaN
        col = (
            df[amenity]
            .astype(str).str.lower().str.strip()
            .map({'true': 1, 'false': 0})
        )
        
        temp = df[[target_col]].copy()
        temp['group'] = col
        temp = temp.dropna(subset=['group', target_col])
        temp['group'] = temp['group'].astype(int)
        
        g1 = temp[temp['group'] == 1][target_col]
        g0 = temp[temp['group'] == 0][target_col]
        
        if len(g1) < 5 or len(g0) < 5:
            continue
        
        # آزمون Mann-Whitney U
        stat, p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
        
        # اندازه اثر Cohen's d
        nx, ny = len(g1), len(g0)
        pooled_std = np.sqrt(((nx - 1) * g1.std()**2 + (ny - 1) * g0.std()**2) / (nx + ny - 2))
        d = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else 0
        
        abs_d = abs(d)
        if abs_d < 0.2:
            effect = 'negligible'
        elif abs_d < 0.5:
            effect = 'small'
        elif abs_d < 0.8:
            effect = 'medium'
        else:
            effect = 'large'
        
        results.append({'amenity': amenity, 'n_with': len(g1), 'n_without': len(g0), 'mean_with': g1.mean(), 
                        'mean_without': g0.mean(), 'median_with': g1.median(), 'median_without': g0.median(), 
                        'p_value': p, 'significant': p < 0.05, 'cohens_d': d, 'effect_size': effect,})
    
    return pd.DataFrame(results).sort_values('p_value')