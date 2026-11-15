# Customer Segmentation using KMeans

This is my project for the Skill Pilot task "Cluster a customer dataset and interpret the segments". I group customers into clusters with KMeans and then try to understand what kind of customers are in each group.

## Project structure

```
customer_segmentation/
  data/
    customers.csv            input data (250 customers)
  src/
    generate_data.py         creates the synthetic dataset
    segment.py               main clustering script
  tests/
    test_pipeline.py         small tests (pytest)
  outputs/
    customers_labeled.csv    data with a Cluster column
    cluster_profile.csv      average of each feature per cluster
    pairplot.png             pair plot coloured by cluster
    k_selection.png          elbow and silhouette plots
    insights.md              short report for each cluster
  requirements.txt
  README.md
```

## Dataset

The dataset is synthetic (made with src/generate_data.py) so the project runs without downloading anything. Columns:

- CustomerID (not used for clustering)
- Age
- AnnualIncome_k (income in thousands)
- SpendingScore (1 to 100)
- AnnualVisits

To use your own data, pass the CSV path to segment.py. It uses all numeric columns automatically and ignores columns with "id" in the name.

## What the code does

1. Loads the CSV with pandas and drops rows with missing values.
2. Scales the numeric columns with StandardScaler. KMeans works on distance, so without scaling a big-range column like income would dominate a small one like visits.
3. Runs KMeans for k = 2 to 8 and calculates inertia (elbow) and silhouette score for each. The k with the highest silhouette score is selected.
4. Fits the final model and adds a Cluster label to every customer.
5. Makes a pair plot coloured by cluster, a cluster profile table, and a short report (insights.md).

## How to run

```
pip install -r requirements.txt
python src/generate_data.py
python src/segment.py
```

With your own file:

```
python src/segment.py path/to/file.csv
```

To run the tests:

```
python -m pytest
```

All results are saved in the outputs folder.

## Results

Best k was 4 with a silhouette score of 0.593.

| Cluster | Customers | Avg Age | Avg Income (k) | Avg Spending Score | Avg Visits |
|---|---|---|---|---|---|
| 0 | 51 | 50.6 | 89.1 | 19.8 | 7.7 |
| 1 | 84 | 32.9 | 42.6 | 36.2 | 13.6 |
| 2 | 60 | 24.4 | 28.5 | 79.6 | 40.0 |
| 3 | 55 | 44.8 | 93.4 | 82.5 | 28.8 |

![Pair plot](outputs/pairplot.png)
![Elbow and silhouette](outputs/k_selection.png)

## Understanding the segments

- Cluster 3 (high income, high spending): these look like the best customers. They earn well, spend a lot and visit often. Loyalty rewards would suit them.
- Cluster 2 (low income, high spending): the youngest group. Income is low but spending score and visits are the highest, so offers and social media promotions could work.
- Cluster 0 (high income, low spending): older customers with good income who spend very little and rarely visit. Better recommendations might get more sales from them.
- Cluster 1 (low income, low spending): the biggest group (34%), everything is on the lower side. Discounts and bundle deals could help.

## Limitations

- The data is synthetic, so the clusters came out clean. Real data would be messier.
- KMeans is affected by outliers and expects roundish clusters.
- I used only four features. Adding things like purchase history could give better segments.

## Possible improvements

- Compare with other algorithms like DBSCAN or Hierarchical clustering.
- Use PCA to see the clusters in 2D.
- Add categorical columns like gender or region.
- Build a small Streamlit dashboard.

Made by Priyanka Raj (MCA)
