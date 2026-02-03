# What Should You Charge for Your Airbnb?

## The Problem

You're hosting a place in NYC. Airbnb's algorithm suggests one price. Your neighbor charges something else. Who's right?

This pricing engine learns from **49,000 real NYC Airbnb listings** (Kaggle dataset) to tell you what similar places actually charge.

## Key Findings

**Location is everything.** A Manhattan studio beats a Brooklyn 2-bedroom every time. Borough matters more than amenities, reviews, or how nice your photos are.

The model predicts prices within ~$25/night accuracy (MAE). Good enough to know if you're way off, not precise enough to optimize down to the dollar.

## What's in Here

- `airbnb_pricing.py` - Full analysis + pricing model
- `airbnb_eda.png` - 6 charts showing what drives prices
- `feature_importance.png` - Which features actually matter
- `model_summary.csv` - Model performance stats

## How to Use

1. Download the dataset: [NYC Airbnb Open Data on Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
2. Save as `AB_NYC_2019.csv` in this folder
3. Run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python airbnb_pricing.py
```

The code will train two models (Random Forest + Gradient Boosting), pick the better one, and show example price recommendations.

## Real-World Use

If I were building this for production:
- Add seasonality (prices spike around holidays)
- Include neighborhood walk scores, subway proximity
- Track how quickly listings get booked at different price points
- Build a simple web interface where hosts input their features and get a price range

The current model is good for "am I in the right ballpark?" Not sophisticated enough for dynamic daily pricing yet.
