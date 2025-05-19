from src.custom_transformer import CustomFeatureProcessor
from src.feature_engineering import scale_encode_data
from sklearn.cluster import KMeans

def add_clusters(df,non_num_cols):
    
    data_transformer = CustomFeatureProcessor(scale_encode_data, non_num_cols=non_num_cols, is_train=True)
    data_transformer.fit(df)
    df_encoded = data_transformer.transform(df)

    kmeans_4 = KMeans(n_clusters= 4, random_state=42)
    kmeans_4.fit(df_encoded)

    # clusters 
    clusters = kmeans_4.predict(df_encoded)

    return clusters