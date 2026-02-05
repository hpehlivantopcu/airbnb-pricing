"""
What Should You Charge for Your Airbnb?
A Pricing Engine for NYC Hosts

Real dataset: Kaggle's "New York City Airbnb Open Data"
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

The problem: You're hosting an apartment in Brooklyn. Airbnb suggests $150/night.
Your neighbor charges $95. Another one gets $220. Who's right?

This builds a pricing model that learns from 49,000 actual listings to tell you:
- What similar places charge
- If you're leaving money on the table
- Which features actually let you charge more

Turns out location matters WAY more than amenities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

print("="*70)
print(" AIRBNB PRICING ENGINE - NYC")
print("="*70)

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

try:
    df = pd.read_csv('AB_NYC_2019.csv')
    print(f"\n[OK] Loaded {len(df):,} Airbnb listings")
except FileNotFoundError:
    print("\n[WARNING] AB_NYC_2019.csv not found!")
    print("   Download from: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data")
    print("   Place it in this folder, then run again.\n")
    exit()

# ═══════════════════════════════════════════════════════════
# CLEAN & PREP
# ═══════════════════════════════════════════════════════════

print("\n-> Cleaning data...")
# Drop listings with missing or extreme prices
df = df[df['price'] > 0]
df = df[df['price'] < 1000]  # Remove outliers
df = df.dropna(subset=['price', 'neighbourhood_group', 'room_type', 
                       'minimum_nights', 'number_of_reviews'])

# Feature engineering
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['availability_rate'] = df['availability_365'] / 365
df['is_active'] = (df['number_of_reviews'] > 0).astype(int)
df['price_per_night'] = df['price']  # target

# Log transform price for modeling
df['log_price'] = np.log1p(df['price'])

print(f"   Clean dataset: {len(df):,} listings")
print(f"   Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
print(f"   Median: ${df['price'].median():.0f}/night")

# ═══════════════════════════════════════════════════════════
# EDA: What Actually Affects Price?
# ═══════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#0d1117')

# 1. Price by borough
borough_price = df.groupby('neighbourhood_group')['price'].median().sort_values(ascending=False)
axes[0,0].barh(range(len(borough_price)), borough_price.values, color='#ff6b6b')
axes[0,0].set_yticks(range(len(borough_price)))
axes[0,0].set_yticklabels(borough_price.index)
axes[0,0].set_xlabel('Median Price ($)')
axes[0,0].set_title('Manhattan is 2x More Expensive', fontweight='bold')
for i, v in enumerate(borough_price.values):
    axes[0,0].text(v + 3, i, f'${v:.0f}', va='center')

# 2. Price by room type
room_price = df.groupby('room_type')['price'].median().sort_values(ascending=False)
axes[0,1].bar(range(len(room_price)), room_price.values, color=['#4ecdc4', '#ffa94d', '#95e1d3'])
axes[0,1].set_xticks(range(len(room_price)))
axes[0,1].set_xticklabels(room_price.index, rotation=20)
axes[0,1].set_ylabel('Median Price ($)')
axes[0,1].set_title('Entire Home = 2x Private Room')
for i, v in enumerate(room_price.values):
    axes[0,1].text(i, v + 3, f'${v:.0f}', ha='center', fontweight='bold')

# 3. Reviews vs Price (does quality matter?)
sampled = df.sample(min(3000, len(df)), random_state=42)
axes[0,2].scatter(sampled['number_of_reviews'], sampled['price'], 
                 alpha=0.3, s=10, color='#ffa94d')
axes[0,2].set_xlabel('Number of Reviews')
axes[0,2].set_ylabel('Price ($)')
axes[0,2].set_title('More Reviews ≠ Higher Price', style='italic')
axes[0,2].set_xlim(0, 200)

# 4. Top 10 neighborhoods by median price
top_hoods = df.groupby('neighbourhood')['price'].agg(['median', 'count'])
top_hoods = top_hoods[top_hoods['count'] >= 20].nlargest(10, 'median')
axes[1,0].barh(range(len(top_hoods)), top_hoods['median'].values, color='#b388ff')
axes[1,0].set_yticks(range(len(top_hoods)))
axes[1,0].set_yticklabels(top_hoods.index, fontsize=9)
axes[1,0].set_xlabel('Median Price ($)')
axes[1,0].set_title('Priciest Neighborhoods')

# 5. Availability patterns
axes[1,1].hist(df['availability_365'], bins=50, color='#4ecdc4', alpha=0.7, edgecolor='black')
axes[1,1].axvline(df['availability_365'].median(), color='#ff6b6b', 
                 linestyle='--', linewidth=2, label=f"Median: {df['availability_365'].median():.0f} days")
axes[1,1].set_xlabel('Days Available per Year')
axes[1,1].set_ylabel('Number of Listings')
axes[1,1].set_title('Availability Distribution')
axes[1,1].legend()

# 6. Minimum nights requirement
min_nights_counts = df['minimum_nights'].value_counts().head(10).sort_index()
axes[1,2].bar(range(len(min_nights_counts)), min_nights_counts.values, color='#95e1d3')
axes[1,2].set_xticks(range(len(min_nights_counts)))
axes[1,2].set_xticklabels(min_nights_counts.index)
axes[1,2].set_xlabel('Minimum Nights Required')
axes[1,2].set_ylabel('Number of Listings')
axes[1,2].set_title('Most Hosts Allow 1-Night Stays')

plt.tight_layout()
plt.savefig('airbnb_eda.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("\n[OK] Saved airbnb_eda.png")

# ═══════════════════════════════════════════════════════════
# BUILD PRICING MODEL
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print(" TRAINING PRICING MODEL")
print("="*70)

# Encode categoricals
le_borough = LabelEncoder()
le_room = LabelEncoder()
df['borough_encoded'] = le_borough.fit_transform(df['neighbourhood_group'])
df['room_encoded'] = le_room.fit_transform(df['room_type'])

# Features for modeling
features = ['borough_encoded', 'room_encoded', 'minimum_nights', 
            'number_of_reviews', 'reviews_per_month', 
            'calculated_host_listings_count', 'availability_365']

X = df[features]
y = df['log_price']  # Predicting log-price for better performance

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Random Forest
print("\n-> Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_pred_price = np.expm1(rf_pred)  # Convert back to actual price
rf_mae = mean_absolute_error(np.expm1(y_test), rf_pred_price)
rf_r2 = r2_score(y_test, rf_pred)
print(f"   MAE: ${rf_mae:.2f}  |  R²: {rf_r2:.3f}")

# Model 2: Gradient Boosting
print("\n-> Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, 
                               max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_pred_price = np.expm1(gb_pred)
gb_mae = mean_absolute_error(np.expm1(y_test), gb_pred_price)
gb_r2 = r2_score(y_test, gb_pred)
print(f"   MAE: ${gb_mae:.2f}  |  R²: {gb_r2:.3f}")

# Pick best
best_model = rf if rf_mae < gb_mae else gb
best_name = "Random Forest" if best_model == rf else "Gradient Boosting"
print(f"\n[OK] Winner: {best_name}")

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
ax.barh(importance['feature'], importance['importance'], color='#4ecdc4')
ax.set_xlabel('Importance', color='white')
ax.set_title('What Actually Matters for Pricing', fontweight='bold', color='white')
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("[OK] Saved feature_importance.png")

# ═══════════════════════════════════════════════════════════
# PRICING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print(" EXAMPLE PRICING RECOMMENDATIONS")
print("="*70)

def get_price_recommendation(borough, room_type, min_nights=1, 
                             num_reviews=0, reviews_pm=0, host_listings=1, avail=365):
    """Get suggested price for a listing"""
    input_data = pd.DataFrame({
        'borough_encoded': [le_borough.transform([borough])[0]],
        'room_encoded': [le_room.transform([room_type])[0]],
        'minimum_nights': [min_nights],
        'number_of_reviews': [num_reviews],
        'reviews_per_month': [reviews_pm],
        'calculated_host_listings_count': [host_listings],
        'availability_365': [avail]
    })
    log_pred = best_model.predict(input_data)[0]
    return np.expm1(log_pred)

# Examples
examples = [
    ("Manhattan", "Entire home/apt", "Luxury 1BR in Midtown"),
    ("Brooklyn", "Private room", "Cozy room in Williamsburg"),
    ("Queens", "Entire home/apt", "2BR near JFK"),
    ("Bronx", "Shared room", "Budget hostel-style"),
]

print("\n{:<50} {:>15}".format("Listing", "Suggested Price"))
print("-" * 70)
for borough, room, desc in examples:
    price = get_price_recommendation(borough, room)
    print(f"{desc:<50} ${price:>14.0f}/night")

# Save model summary
summary = {
    'model': [best_name],
    'mae': [gb_mae if best_model == gb else rf_mae],
    'r2': [gb_r2 if best_model == gb else rf_r2],
    'train_size': [len(X_train)],
    'test_size': [len(X_test)]
}
pd.DataFrame(summary).to_csv('model_summary.csv', index=False)
print("\n[OK] Saved model_summary.csv")

print("\n" + "="*70)
print(" DONE - Check the PNG files for visualizations")
print("="*70)
