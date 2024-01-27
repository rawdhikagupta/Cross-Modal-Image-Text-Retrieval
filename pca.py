import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Assuming 'train_features_subset' contains the extracted features
# Shape of train_features_subset: (num_images, feature_dim)

# Apply PCA to reduce the dimensionality to 2 for visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(train_features_subset.numpy())

# Create a scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=image_ids_subset, palette="viridis", legend="full")
plt.title('PCA Visualization of ViT Features')
plt.show()
