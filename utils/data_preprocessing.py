import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Check and handle missing values
    data.fillna(data.median(), inplace=True)

    # Split features and target
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Save original feature names before scaling
    feature_names = X.columns

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def apply_pca(X_train, X_test, n_components=2):
    """
    Apply Principal Component Analysis (PCA) to reduce dimensions.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Explained Variance Ratio (PCA): {pca.explained_variance_ratio_}")

    return X_train_pca, X_test_pca

def plot_correlation_matrix(data):
    """
    Plot an improved and clear correlation matrix for the dataset.
    """
    correlation = data.corr()

    plt.figure(figsize=(10, 8))  # Set figure size for better readability
    heatmap = sns.heatmap(
        correlation,
        annot=True,              # Display correlation coefficients
        fmt=".2f",               # Format coefficients to 2 decimal places
        cmap="coolwarm",         # Use a color palette that highlights differences
        square=True,             # Keep cells square-shaped
        cbar_kws={'shrink': 0.8}, # Shrink the color bar for aesthetics
        linewidths=0.5,          # Add grid lines between cells
    )

    # Add title and labels
    heatmap.set_title("Correlation Matrix", fontdict={'fontsize': 16}, pad=20)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")  # Save the plot as an image file
    plt.close()  # Close the plot to prevent overlapping in successive calls
