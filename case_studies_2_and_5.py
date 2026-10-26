import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.prop_cycle": cycler(color=["black", "dimgray", "gray", "darkgray"]),
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "lightgray",
})

strength_map = LinearSegmentedColormap.from_list("strength", ["#8c8c8c", "#ffffff", "#8c8c8c"])

airline = pd.read_excel("data/EastWestAirlines.xlsx", sheet_name="data")
print(airline.shape)
print(airline.head())
airline.info()

airline = airline.drop(columns="ID#")
print(airline.isna().sum())
print("Duplicate rows:", airline.duplicated().sum())
airline = airline.drop_duplicates().reset_index(drop=True)
print(airline.shape)

print(airline.describe().T.round(2))

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), ["Balance", "Bonus_miles", "Flight_miles_12mo", "Days_since_enroll"]):
    ax.hist(airline[col], bins=40, color="gray", edgecolor="black")
    ax.set_title("Distribution of " + col)
    ax.set_xlabel(col)
    ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.show()

numeric_cols = ["Balance", "Bonus_miles", "Bonus_trans", "Flight_miles_12mo", "Flight_trans_12"]
fig, axes = plt.subplots(1, 5, figsize=(14, 4.5))
for ax, col in zip(axes, numeric_cols):
    ax.boxplot(airline[col], patch_artist=True,
               boxprops=dict(facecolor="lightgray", edgecolor="black"),
               medianprops=dict(color="black"),
               flierprops=dict(marker="o", markerfacecolor="gray", markeredgecolor="black", markersize=3))
    ax.set_title(col)
    ax.set_xticks([])
    ax.set_ylabel("Value")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for ax, col in zip(axes, ["cc1_miles", "cc2_miles", "cc3_miles", "Award?"]):
    counts = airline[col].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="gray", edgecolor="black")
    ax.set_title("Count of " + col)
    ax.set_xlabel(col)
    ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(airline["Bonus_miles"], airline["Balance"], s=8, color="gray", alpha=0.6)
axes[0].set_title("Balance vs Bonus Miles")
axes[0].set_xlabel("Bonus miles")
axes[0].set_ylabel("Balance (miles)")
axes[1].scatter(airline["Flight_trans_12"], airline["Flight_miles_12mo"], s=8, color="gray", alpha=0.6)
axes[1].set_title("Flight Miles vs Flight Transactions")
axes[1].set_xlabel("Flight transactions in 12 months")
axes[1].set_ylabel("Flight miles in 12 months")
plt.tight_layout()
plt.show()

corr_air = airline.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_air, annot=True, fmt=".2f", cmap=strength_map, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="black", cbar_kws={"label": "Correlation (shade shows strength)"},
            annot_kws={"color": "black", "size": 8})
plt.title("Correlation Heatmap of Airline Customer Variables")
plt.tight_layout()
plt.show()

outlier_cols = ["Balance", "Bonus_miles", "Bonus_trans", "Flight_miles_12mo", "Flight_trans_12"]
z_scores = (airline[outlier_cols] - airline[outlier_cols].mean()) / airline[outlier_cols].std()
keep = (z_scores.abs() <= 3).all(axis=1)
print("Rows before:", len(airline))
print("Rows removed:", (~keep).sum())
airline_clean = airline[keep].reset_index(drop=True)
print("Rows after:", len(airline_clean))

scaler = StandardScaler()
X_air = scaler.fit_transform(airline_clean)
print(pd.DataFrame(X_air, columns=airline_clean.columns).describe().loc[["mean", "std"]].round(2))

k_values = list(range(2, 11))
inertias = []
sil_scores = []
for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_air)
    inertias.append(model.inertia_)
    sil_scores.append(silhouette_score(X_air, model.labels_))
print(pd.DataFrame({"k": k_values, "inertia": np.round(inertias, 1), "silhouette": np.round(sil_scores, 3)}))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(k_values, inertias, marker="o", color="black")
axes[0].set_title("Elbow Curve for K-Means")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)
axes[1].plot(k_values, sil_scores, marker="s", color="black")
axes[1].set_title("Silhouette Score for Different k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].grid(True)
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_air)
labels_km = kmeans.labels_
sil_km = silhouette_score(X_air, labels_km)
print("K-Means silhouette score:", round(sil_km, 3))
print(pd.Series(labels_km).value_counts().sort_index())

Z = linkage(X_air, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30, color_threshold=0, above_threshold_color="black")
plt.title("Dendrogram (Ward Linkage, Last 30 Merged Clusters)")
plt.xlabel("Cluster size or sample index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

rows = []
for method in ["ward", "complete", "average", "single"]:
    for k in [3, 4, 5]:
        lab = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X_air)
        sizes = np.bincount(lab)
        rows.append([method, k, round(silhouette_score(X_air, lab), 3), int(sizes.max()), int(sizes.min())])
linkage_table = pd.DataFrame(rows, columns=["linkage", "clusters", "silhouette", "largest_cluster", "smallest_cluster"])
print(linkage_table)

hier = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X_air)
labels_hc = hier.labels_
sil_hc = silhouette_score(X_air, labels_hc)
print("Hierarchical (Ward) silhouette score:", round(sil_hc, 3))
print(pd.Series(labels_hc).value_counts().sort_index())

neighbors = NearestNeighbors(n_neighbors=10).fit(X_air)
distances, _ = neighbors.kneighbors(X_air)
k_dist = np.sort(distances[:, -1])
plt.figure(figsize=(8, 4.5))
plt.plot(k_dist, color="black")
plt.axhline(1.5, color="black", linestyle="--", linewidth=1, label="Chosen eps = 1.5")
plt.ylim(0, 6)
plt.legend()
plt.title("K-Distance Plot (10th Nearest Neighbour)")
plt.xlabel("Points sorted by distance")
plt.ylabel("Distance to 10th nearest neighbour")
plt.grid(True)
plt.tight_layout()
plt.show()

rows = []
for eps in [0.8, 1.0, 1.2, 1.5, 2.0]:
    for min_pts in [5, 10, 20]:
        lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X_air)
        n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
        noise = int((lab == -1).sum())
        sil = silhouette_score(X_air[lab != -1], lab[lab != -1]) if n_clusters > 1 else np.nan
        rows.append([eps, min_pts, n_clusters, noise, round(sil, 3)])
dbscan_table = pd.DataFrame(rows, columns=["eps", "minPts", "clusters", "noise_points", "silhouette"])
print(dbscan_table)

dbscan = DBSCAN(eps=1.5, min_samples=10).fit(X_air)
labels_db = dbscan.labels_
mask_db = labels_db != -1
sil_db = silhouette_score(X_air[mask_db], labels_db[mask_db])
print("DBSCAN silhouette score (without noise):", round(sil_db, 3))
print(pd.Series(labels_db).value_counts().sort_index())

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_air)
print("Variance explained:", np.round(pca.explained_variance_ratio_, 3))

shades = ["black", "dimgray", "gray", "darkgray", "lightgray"]
markers = ["o", "s", "^", "D", "v"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, labs, title in zip(axes, [labels_km, labels_hc, labels_db],
                           ["K-Means (k = 4)", "Hierarchical (Ward, 4 clusters)", "DBSCAN (eps = 1.5, minPts = 10)"]):
    for i, c in enumerate(sorted(set(labs))):
        pts = X_pca[labs == c]
        if c == -1:
            ax.scatter(pts[:, 0], pts[:, 1], marker="x", color="black", s=14, label="Noise")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], marker=markers[i % 5], facecolor=shades[i % 5],
                       edgecolor="black", linewidth=0.4, s=16, label="Cluster " + str(c))
    ax.set_title(title)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

profile_df = airline_clean.copy()
profile_df["KMeans"] = labels_km
profile_df["Hierarchical"] = labels_hc
profile_df["DBSCAN"] = labels_db

km_profile = profile_df.groupby("KMeans").mean().drop(columns=["Hierarchical", "DBSCAN"]).round(1)
km_profile.insert(0, "Customers", profile_df["KMeans"].value_counts().sort_index())
print(km_profile.T)

hc_profile = profile_df.groupby("Hierarchical").mean().drop(columns=["KMeans", "DBSCAN"]).round(1)
hc_profile.insert(0, "Customers", profile_df["Hierarchical"].value_counts().sort_index())
print(hc_profile.T)

db_profile = profile_df.groupby("DBSCAN").mean().drop(columns=["KMeans", "Hierarchical"]).round(1)
db_profile.insert(0, "Customers", profile_df["DBSCAN"].value_counts().sort_index())
print(db_profile.T)

features = ["Balance", "Bonus_miles", "Flight_miles_12mo", "Flight_trans_12", "cc1_miles", "Days_since_enroll"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.ravel(), features):
    ax.bar(km_profile.index.astype(str), km_profile[col], color="gray", edgecolor="black")
    ax.set_title("Average " + col)
    ax.set_xlabel("K-Means cluster")
    ax.set_ylabel("Average value")
plt.tight_layout()
plt.show()

summary = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical (Ward)", "DBSCAN"],
    "Clusters": [4, 4, len(set(labels_db)) - 1],
    "Noise points": [0, 0, int((labels_db == -1).sum())],
    "Silhouette score": [round(sil_km, 3), round(sil_hc, 3), round(sil_db, 3)],
})
print(summary)

red = pd.read_csv("data/winequality-red.csv", sep=";")
white = pd.read_csv("data/winequality-white.csv", sep=";")
print("Red wine:", red.shape)
print("White wine:", white.shape)
wine = pd.concat([red, white], ignore_index=True)
print("Combined:", wine.shape)
print(wine.head())
wine.info()
print(wine.isna().sum().sum())
print("Duplicate rows:", wine.duplicated().sum())

selected = ["alcohol", "volatile acidity", "residual sugar"]
desc = pd.DataFrame({
    "Mean": wine[selected].mean(),
    "Median": wine[selected].median(),
    "Mode": [wine[c].mode().iloc[0] for c in selected],
    "Std Dev": wine[selected].std(),
    "Skewness": wine[selected].skew(),
})
print(desc.round(4))

cv = (wine[selected].std() / wine[selected].mean() * 100).round(2)
print(cv)
print("Highest relative variability:", cv.idxmax())

r_value, p_value = stats.pearsonr(wine["alcohol"], wine["quality"])
print("Pearson r:", round(r_value, 4))
print("p-value:", p_value)

print(wine["quality"].value_counts().sort_index())
plt.figure(figsize=(8, 4.5))
plt.hist(wine["quality"], bins=np.arange(2.5, 10.5, 1), color="gray", edgecolor="black")
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality score")
plt.ylabel("Number of wines")
plt.xticks(range(3, 10))
plt.tight_layout()
plt.show()

levels = sorted(wine["quality"].unique())
groups = [wine.loc[wine["quality"] == q, "alcohol"] for q in levels]
plt.figure(figsize=(9, 5))
plt.boxplot(groups, patch_artist=True,
            boxprops=dict(facecolor="lightgray", edgecolor="black"),
            medianprops=dict(color="black"),
            flierprops=dict(marker="o", markerfacecolor="gray", markeredgecolor="black", markersize=4))
plt.xticks(range(1, len(levels) + 1), levels)
plt.title("Alcohol Content by Wine Quality Score")
plt.xlabel("Quality score")
plt.ylabel("Alcohol (% by volume)")
plt.tight_layout()
plt.show()
print(wine.groupby("quality")["alcohol"].agg(["count", "median", "mean"]).round(2))

corr_wine = wine.corr()
plt.figure(figsize=(11, 8.5))
sns.heatmap(corr_wine, annot=True, fmt=".2f", cmap=strength_map, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="black", cbar_kws={"label": "Correlation (shade shows strength)"},
            annot_kws={"color": "black", "size": 8})
plt.title("Correlation Heatmap of Physicochemical Attributes and Quality")
plt.tight_layout()
plt.show()

quality_corr = corr_wine["quality"].drop("quality")
ranked = quality_corr.reindex(quality_corr.abs().sort_values(ascending=False).index)
print(ranked.round(4))

X = wine.drop(columns="quality")
y = wine["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

lin_reg = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))

errors = np.abs(y_test - y_pred)
print("Predictions within 0.5 of actual:", round((errors <= 0.5).mean() * 100, 1), "%")
print("Predictions within 1.0 of actual:", round((errors <= 1.0).mean() * 100, 1), "%")
print("Predicted range:", round(y_pred.min(), 2), "to", round(y_pred.max(), 2))

std_coef = pd.Series(lin_reg.coef_ * X_train.std().values / y_train.std(), index=X.columns)
std_coef_sorted = std_coef.reindex(std_coef.abs().sort_values(ascending=False).index)
print(pd.DataFrame({"Raw coefficient": pd.Series(lin_reg.coef_, index=X.columns).round(4),
                    "Standardised coefficient": std_coef.round(4)}).loc[std_coef_sorted.index])

plt.figure(figsize=(9, 5))
plt.barh(std_coef_sorted.index[::-1], std_coef_sorted.values[::-1], color="gray", edgecolor="black")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Standardised Regression Coefficients")
plt.xlabel("Standardised coefficient")
plt.ylabel("Attribute")
plt.tight_layout()
plt.show()

vif_values = {}
for col in X_train.columns:
    others = X_train.drop(columns=col)
    r2_col = LinearRegression().fit(others, X_train[col]).score(others, X_train[col])
    vif_values[col] = 1 / (1 - r2_col)
vif = pd.Series(vif_values).sort_values(ascending=False)
print(vif.round(2))

plt.figure(figsize=(7, 6))
rng = np.random.default_rng(42)
jitter = rng.uniform(-0.15, 0.15, size=len(y_test))
plt.scatter(y_test + jitter, y_pred, s=10, color="gray", alpha=0.5, edgecolor="none")
plt.plot([3, 9], [3, 9], color="black", linestyle="--", label="Perfect prediction")
plt.title("Predicted vs Actual Wine Quality (Test Set)")
plt.xlabel("Actual quality score (small jitter added)")
plt.ylabel("Predicted quality score")
plt.legend()
plt.tight_layout()
plt.show()

rows = []
for name, data in [("Red", red), ("White", white), ("Combined", wine)]:
    Xs = data.drop(columns="quality")
    ys = data["quality"]
    Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.2, random_state=42)
    pred = LinearRegression().fit(Xa, ya).predict(Xb)
    rows.append([name, len(data), round(mean_absolute_error(yb, pred), 3),
                 round(np.sqrt(mean_squared_error(yb, pred)), 3), round(r2_score(yb, pred), 3)])
print(pd.DataFrame(rows, columns=["Dataset", "Rows", "MAE", "RMSE", "R2"]))
