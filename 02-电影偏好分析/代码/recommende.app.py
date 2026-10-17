#!/usr/bin/env python3
"""
电影偏好分析系统 - 增强版
Movie Preference Analysis System - Enhanced Version

功能：
1. 支持新的数据格式（用户id、movie_id、评分、电影名称、类型、导演、时长、票房、国家/地区）
2. 客户群体分析（K-Means聚类）
3. 电影类型偏好预测（多标签分类）
4. 协同过滤推荐算法
5. 数据可视化
6. 模型评估和对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CollaborativeFiltering:
    """协同过滤推荐系统"""
    def __init__(self):
        self.user_movie_matrix = None
        self.user_similarity_matrix = None
        self.movie_similarity_matrix = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

    def fit(self, ratings_data):
        print("🔍 训练协同过滤模型...")
        user_ids = self.user_encoder.fit_transform(ratings_data['user_id'])
        movie_ids = self.movie_encoder.fit_transform(ratings_data['movie_id'])
        self.user_movie_matrix = csr_matrix(
            (ratings_data['评分'], (user_ids, movie_ids)),
            shape=(len(self.user_encoder.classes_), len(self.movie_encoder.classes_))
        ).toarray()
        self.user_similarity_matrix = cosine_similarity(self.user_movie_matrix)
        self.movie_similarity_matrix = cosine_similarity(self.user_movie_matrix.T)
        print(f"✅ 协同过滤模型训练完成")
        print(f"   用户数量: {len(self.user_encoder.classes_)}")
        print(f"   电影数量: {len(self.movie_encoder.classes_)}")

    def get_user_recommendations(self, user_id, n_recommendations=10, method='user_based'):
        if method == 'user_based':
            return self._user_based_recommendations(user_id, n_recommendations)
        else:
            return self._item_based_recommendations(user_id, n_recommendations)

    def _user_based_recommendations(self, user_id, n_recommendations):
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            user_ratings = self.user_movie_matrix[user_idx]
            unwatched_movies = np.where(user_ratings == 0)[0]
            if len(unwatched_movies) == 0:
                return []
            user_similarities = self.user_similarity_matrix[user_idx]
            predictions = []
            for movie_idx in unwatched_movies:
                movie_ratings = self.user_movie_matrix[:, movie_idx]
                rated_users = np.where(movie_ratings > 0)[0]
                if len(rated_users) == 0:
                    continue
                similarities = user_similarities[rated_users]
                ratings = movie_ratings[rated_users]
                valid_mask = similarities > 0
                if not valid_mask.any():
                    continue
                similarities = similarities[valid_mask]
                ratings = ratings[valid_mask]
                predicted_rating = np.sum(similarities * ratings) / np.sum(similarities)
                predictions.append((movie_idx, predicted_rating))
            predictions.sort(key=lambda x: x[1], reverse=True)
            recommended_movies = []
            for movie_idx, predicted_rating in predictions[:n_recommendations]:
                movie_id = self.movie_encoder.inverse_transform([movie_idx])[0]
                recommended_movies.append({
                    'movie_id': movie_id,
                    'predicted_rating': predicted_rating
                })
            return recommended_movies
        except Exception as e:
            print(f"用户推荐出错: {e}")
            return []

    def _item_based_recommendations(self, user_id, n_recommendations):
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            user_ratings = self.user_movie_matrix[user_idx]
            rated_movies = np.where(user_ratings > 0)[0]
            if len(rated_movies) == 0:
                return []
            unwatched_movies = np.where(user_ratings == 0)[0]
            if len(unwatched_movies) == 0:
                return []
            predictions = []
            for movie_idx in unwatched_movies:
                movie_similarities = self.movie_similarity_matrix[movie_idx, rated_movies]
                valid_mask = movie_similarities > 0
                if not valid_mask.any():
                    continue
                similarities = movie_similarities[valid_mask]
                ratings = user_ratings[rated_movies[valid_mask]]
                predicted_rating = np.sum(similarities * ratings) / np.sum(similarities)
                predictions.append((movie_idx, predicted_rating))
            predictions.sort(key=lambda x: x[1], reverse=True)
            recommended_movies = []
            for movie_idx, predicted_rating in predictions[:n_recommendations]:
                movie_id = self.movie_encoder.inverse_transform([movie_idx])[0]
                recommended_movies.append({
                    'movie_id': movie_id,
                    'predicted_rating': predicted_rating
                })
            return recommended_movies
        except Exception as e:
            print(f"物品推荐出错: {e}")
            return []

class MoviePreferenceAnalyzer:
    """电影偏好分析器"""
    def __init__(self):
        self.data = None
        self.movies_data = None
        self.ratings_data = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.classification_models = {}
        self.cluster_labels = None
        self.feature_columns = []
        self.target_columns = []
        self.collaborative_filtering = CollaborativeFiltering()

    def load_data(self, file_path=None):
        if file_path and file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
            self._process_data()
        else:
            self.data = self._generate_sample_data()
        print(f"✅ 数据加载完成，共 {len(self.data)} 条记录")
        print(f"📊 数据列：{list(self.data.columns)}")
        return self.data

    def _process_data(self):
        self.movies_data = self.data[['movie_id', '电影名称', '类型', '导演', '时长', '票房', '国家/地区']].drop_duplicates()
        self.ratings_data = self.data[['user_id', 'movie_id', '评分']].copy()
        self._process_genre_column()
        print(f"✅ 数据处理完成")
        print(f"   电影数量: {len(self.movies_data)}")
        print(f"   评分数量: {len(self.ratings_data)}")
        print(f"   用户数量: {self.ratings_data['user_id'].nunique()}")

    def _process_genre_column(self):
        all_genres = set()
        for genres in self.movies_data['类型']:
            if isinstance(genres, str):
                all_genres.update(genres.split('-'))
        for genre in all_genres:
            self.movies_data[f'genre_{genre}'] = self.movies_data['类型'].apply(
                lambda x: 1 if isinstance(x, str) and genre in x.split('-') else 0
            )
        self.genre_list = [f'genre_{genre}' for genre in all_genres]
        print(f"   电影类型: {list(all_genres)}")

    def _generate_sample_data(self, n_samples=10000):
        np.random.seed(42)
        n_movies = 500
        movie_ids = range(1, n_movies + 1)
        movie_names = [
            '肖申克的救赎', '教父', '教父2', '黑暗骑士', '12怒汉', '辛德勒的名单',
            '指环王：王者归来', '低俗小说', '好家伙', '搏击俱乐部', '指环王：护戒使者',
            '星球大战：帝国反击战', '阿甘正传', '盗梦空间', '指环王：双塔奇兵',
            '星球大战：新希望', '黑客帝国', '好家伙', '七武士', '沉默的羔羊',
            '指环王：护戒使者', '星球大战：绝地归来', '阿凡达', '泰坦尼克号',
            '复仇者联盟', '速度与激情', '碟中谍', '007', '变形金刚', '功夫',
            '夏洛特烦恼', '疯狂动物城', '神偷奶爸', '功夫熊猫', '爱乐之城',
            '情书', '罗马假日', '乱世佳人', '闪灵', '寂静岭', '咒怨', '午夜凶铃'
        ]
        while len(movie_names) < n_movies:
            movie_names.extend([f"电影{i}" for i in range(len(movie_names), n_movies)])
        directors = ['斯皮尔伯格', '诺兰', '卡梅隆', '昆汀', '科波拉', '斯科塞斯',
                    '芬奇', '库布里克', '希区柯克', '黑泽明', '张艺谋', '李安',
                    '王家卫', '周星驰', '徐峥', '宁浩', '冯小刚', '陈凯歌']
        genres = ['动作', '喜剧', '剧情', '科幻', '恐怖', '爱情', '悬疑', '动画', '战争', '西部']
        countries = ['美国', '中国', '日本', '韩国', '英国', '法国', '德国', '意大利', '印度', '俄罗斯']
        movies_data = []
        for i in range(n_movies):
            movie_genres = np.random.choice(genres, np.random.randint(1, 4), replace=False)
            genre_str = '-'.join(movie_genres)
            country = np.random.choice(countries)
            duration = np.random.randint(90, 181)
            boxoffice = np.random.randint(1000000, 1000000000)
            movies_data.append({
                'movie_id': movie_ids[i],
                '电影名称': movie_names[i],
                '类型': genre_str,
                '导演': np.random.choice(directors),
                '时长': duration,
                '票房': boxoffice,
                '国家/地区': country
            })
        n_users = 1000
        ratings_data = []
        for user_id in range(1, n_users + 1):
            n_watched = np.random.randint(10, 51)
            watched_movies = np.random.choice(movie_ids, n_watched, replace=False)
            for movie_id in watched_movies:
                base_rating = np.random.normal(7.0, 1.5)
                rating = np.clip(base_rating, 1, 10)
                ratings_data.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    '评分': round(rating, 1)
                })
        self.movies_data = pd.DataFrame(movies_data)
        self.ratings_data = pd.DataFrame(ratings_data)
        self._process_genre_column()
        self.data = self.ratings_data.merge(self.movies_data, on='movie_id', how='left')
        return self.data

    def prepare_data(self):
        if self.data is None:
            print("❌ 请先加载数据")
            return
        user_features = self.ratings_data.groupby('user_id').agg({
            '评分': ['mean', 'count', 'std'],
            'movie_id': 'nunique'
        }).round(2)
        user_features.columns = ['avg_rating', 'total_ratings', 'rating_std', 'movies_watched']
        user_features = user_features.reset_index()
        user_genre_preferences = self._calculate_user_genre_preferences()
        self.user_features = user_features.merge(user_genre_preferences, on='user_id', how='left')
        self.feature_columns = [
            'avg_rating', 'total_ratings', 'rating_std', 'movies_watched'
        ] + [col for col in self.user_features.columns if col.startswith('genre_')]
        self.target_columns = [col for col in self.user_features.columns if col.startswith('genre_')]
        print(f"✅ 数据准备完成")
        print(f"📊 特征列：{len(self.feature_columns)} 个")
        print(f"🎯 目标列：{len(self.target_columns)} 个")

    def _calculate_user_genre_preferences(self):
        user_movie_data = self.ratings_data.merge(self.movies_data, on='movie_id', how='left')
        genre_preferences = {}
        for genre_col in self.genre_list:
            genre_name = genre_col.replace('genre_', '')
            genre_movies = user_movie_data[user_movie_data[genre_col] == 1]
            if not genre_movies.empty:
                user_genre_ratings = genre_movies.groupby('user_id')['评分'].mean()
                genre_preferences[f'avg_{genre_name}_rating'] = user_genre_ratings
            else:
                genre_preferences[f'avg_{genre_name}_rating'] = pd.Series(0, index=user_movie_data['user_id'].unique())
        user_genre_prefs = pd.DataFrame(genre_preferences).reset_index()
        user_genre_prefs.columns = ['user_id'] + list(genre_preferences.keys())
        return user_genre_prefs

    def train_collaborative_filtering(self):
        print("\n🤝 训练协同过滤推荐模型...")
        self.collaborative_filtering.fit(self.ratings_data)
        print("✅ 协同过滤模型训练完成")

    def get_recommendations(self, user_id, n_recommendations=10, method='user_based'):
        if not hasattr(self.collaborative_filtering, 'user_movie_matrix'):
            print("❌ 请先训练协同过滤模型")
            return []
        print(f"\n🎬 为用户 {user_id} 生成推荐 (方法: {method}):")
        recommendations = self.collaborative_filtering.get_user_recommendations(
            user_id, n_recommendations, method
        )
        if not recommendations:
            print("❌ 无法生成推荐")
            return []
        recommended_movies = []
        for rec in recommendations:
            movie_id = rec['movie_id']
            movie_info = self.movies_data[self.movies_data['movie_id'] == movie_id]
            if not movie_info.empty:
                movie_info = movie_info.iloc[0]
                recommended_movies.append({
                    'movie_id': movie_id,
                    '电影名称': movie_info['电影名称'],
                    '类型': movie_info['类型'],
                    '导演': movie_info['导演'],
                    '时长': movie_info['时长'],
                    '票房': movie_info['票房'],
                    '国家/地区': movie_info['国家/地区'],
                    '预测评分': round(rec['predicted_rating'], 2)
                })
        print(f"\n📋 推荐电影 (共 {len(recommended_movies)} 部):")
        for i, movie in enumerate(recommended_movies, 1):
            print(f"   {i}. {movie['电影名称']} ({movie['类型']})")
            print(f"      导演: {movie['导演']} | 时长: {movie['时长']}分钟")
            print(f"      票房: {movie['票房']:,} | 国家: {movie['国家/地区']}")
            print(f"      预测评分: {movie['预测评分']}")
            print()
        return recommended_movies

    def customer_segmentation(self, n_clusters=4):
        """客户群体分析（K-Means聚类）"""
        print("\n🔍 开始客户群体分析...")
        if not hasattr(self, 'user_features'):
            print("❌ 请先准备数据")
            return
        # 准备聚类特征
        cluster_features = self.user_features[self.feature_columns].copy()
        # 处理缺失值
        cluster_features = cluster_features.fillna(0)
        # 标准化
        cluster_features_scaled = self.scaler.fit_transform(cluster_features)
        # K-Means聚类
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(cluster_features_scaled)
        # 添加聚类标签到数据中
        self.user_features['cluster'] = self.cluster_labels
        # 分析聚类结果
        self._analyze_clusters()
        # 可视化聚类结果
        self._visualize_clusters(cluster_features_scaled)
        print(f"✅ 客户群体分析完成，共分为 {n_clusters} 个群体")

    def _analyze_clusters(self):
        """分析聚类结果"""
        cluster_analysis = self.user_features.groupby('cluster').agg({
            'avg_rating': ['mean', 'std'],
            'total_ratings': ['mean', 'std'],
            'movies_watched': ['mean', 'std']
        }).round(2)
        print("\n📊 聚类群体特征分析：")
        print(cluster_analysis)
        # 各群体的电影类型偏好
        genre_cols = [col for col in self.user_features.columns if col.startswith('avg_') and col.endswith('_rating')]
        if genre_cols:
            genre_preferences = self.user_features.groupby('cluster')[genre_cols].mean()
            print("\n🎬 各群体电影类型偏好：")
            print(genre_preferences.round(3))

    def _visualize_clusters(self, features_scaled):
        """可视化聚类结果"""
        # PCA降维用于可视化
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        plt.figure(figsize=(15, 5))
        # 聚类散点图
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                            c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.title('客户群体聚类结果')
        plt.xlabel('主成分1')
        plt.ylabel('主成分2')
        plt.colorbar(scatter)
        # 平均评分分布
        plt.subplot(1, 3, 2)
        for cluster in sorted(self.user_features['cluster'].unique()):
            cluster_data = self.user_features[self.user_features['cluster'] == cluster]
            plt.hist(cluster_data['avg_rating'], alpha=0.6, label=f'群体{cluster}', bins=20)
        plt.title('各群体平均评分分布')
        plt.xlabel('平均评分')
        plt.ylabel('频次')
        plt.legend()
        # 观影数量分布
        plt.subplot(1, 3, 3)
        for cluster in sorted(self.user_features['cluster'].unique()):
            cluster_data = self.user_features[self.user_features['cluster'] == cluster]
            plt.hist(cluster_data['movies_watched'], alpha=0.6, label=f'群体{cluster}', bins=20)
        plt.title('各群体观影数量分布')
        plt.xlabel('观影数量')
        plt.ylabel('频次')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train_classification_models(self):
        """训练分类模型"""
        print("\n🤖 开始训练分类模型...")
        if not hasattr(self, 'user_features'):
            print("❌ 请先准备数据")
            return
        # 准备数据
        X = self.user_features[self.feature_columns].fillna(0)
        # 为每种类型创建偏好标签（基于平均评分）
        y_data = {}
        for genre_col in self.genre_list:
            genre_name = genre_col.replace('genre_', '')
            avg_rating_col = f'avg_{genre_name}_rating'
            if avg_rating_col in self.user_features.columns:
                # 如果用户对该类型的平均评分大于整体平均，则认为喜欢
                threshold = self.user_features[avg_rating_col].mean()
                y_data[genre_name] = (self.user_features[avg_rating_col] > threshold).astype(int)
        y = pd.DataFrame(y_data)
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # 定义模型
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(kernel='rbf', random_state=42)
        }
        # 训练和评估模型
        results = {}
        for name, model in models.items():
            print(f"\n📈 训练 {name}...")
            # 使用多标签分类器
            multi_model = MultiOutputClassifier(model)
            multi_model.fit(X_train_scaled, y_train)
            # 预测
            y_pred = multi_model.predict(X_test_scaled)
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(multi_model, X_train_scaled, y_train, cv=5)
            results[name] = {
                'model': multi_model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            print(f"   准确率: {accuracy:.4f}")
            print(f"   交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        self.classification_models = results
        self.X_test = X_test
        self.y_test = y_test
        print(f"\n✅ 分类模型训练完成")
        return results

    def compare_models(self):
        """比较模型性能"""
        if not self.classification_models:
            print("❌ 请先训练模型")
            return
        print("\n📊 模型性能对比：")
        # 创建对比表格
        comparison_data = []
        for name, result in self.classification_models.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'CV Mean': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}"
            })
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        # 可视化对比
        self._visualize_model_comparison()

    def _visualize_model_comparison(self):
        """可视化模型对比"""
        models = list(self.classification_models.keys())
        accuracies = [self.classification_models[name]['accuracy'] for name in models]
        cv_means = [self.classification_models[name]['cv_mean'] for name in models]
        cv_stds = [self.classification_models[name]['cv_std'] for name in models]
        plt.figure(figsize=(12, 5))
        # 准确率对比
        plt.subplot(1, 2, 1)
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('模型准确率对比')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        # 交叉验证结果
        plt.subplot(1, 2, 2)
        x_pos = np.arange(len(models))
        plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5,
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('交叉验证结果')
        plt.ylabel('准确率')
        plt.xticks(x_pos, models)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_genre_preferences(self):
        """分析电影类型偏好"""
        print("\n🎬 分析电影类型偏好...")
        if not hasattr(self, 'user_features'):
            print("❌ 请先准备数据")
            return
        # 整体偏好
        genre_cols = [col for col in self.user_features.columns if col.startswith('avg_') and col.endswith('_rating')]
        if not genre_cols:
            print("❌ 没有找到类型偏好数据")
            return
        overall_preferences = self.user_features[genre_cols].mean().sort_values(ascending=False)
        plt.figure(figsize=(15, 10))
        # 整体偏好
        plt.subplot(2, 3, 1)
        overall_preferences.plot(kind='bar', color='skyblue')
        plt.title('整体电影类型偏好')
        plt.ylabel('平均评分')
        plt.xticks(rotation=45)
        # 按聚类群体分析
        plt.subplot(2, 3, 2)
        cluster_preferences = self.user_features.groupby('cluster')[genre_cols].mean()
        cluster_preferences.plot(kind='bar')
        plt.title('按聚类群体分析偏好')
        plt.ylabel('平均评分')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # 评分分布
        plt.subplot(2, 3, 3)
        plt.hist(self.user_features['avg_rating'], bins=30, alpha=0.7, color='lightgreen')
        plt.title('用户平均评分分布')
        plt.xlabel('平均评分')
        plt.ylabel('用户数量')
        # 观影数量分布
        plt.subplot(2, 3, 4)
        plt.hist(self.user_features['movies_watched'], bins=30, alpha=0.7, color='salmon')
        plt.title('用户观影数量分布')
        plt.xlabel('观影数量')
        plt.ylabel('用户数量')
        # 偏好相关性热图
        plt.subplot(2, 3, 5)
        correlation_matrix = self.user_features[genre_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('电影类型偏好相关性')
        # 用户特征与偏好关系
        plt.subplot(2, 3, 6)
        feature_preference_corr = self.user_features[self.feature_columns + genre_cols].corr()
        # 只显示特征与偏好的相关性
        feature_pref_corr = feature_preference_corr.loc[self.feature_columns, genre_cols]
        sns.heatmap(feature_pref_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('用户特征与偏好关系')
        plt.tight_layout()
        plt.savefig('genre_preferences.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 电影类型偏好分析完成")
        print(f"📊 最受欢迎的类型: {overall_preferences.index[0]} ({overall_preferences.iloc[0]:.3f})")

    def run_full_analysis(self, data_path=None):
        print("🚀 开始电影偏好分析系统")
        print("=" * 50)
        self.load_data(data_path)
        self.prepare_data()
        self.train_collaborative_filtering()
        self.customer_segmentation()
        self.train_classification_models()
        self.compare_models()
        self.analyze_genre_preferences()
        if hasattr(self, 'user_features') and len(self.user_features) > 0:
            example_user = self.user_features['user_id'].iloc[0]
            self.get_recommendations(example_user, 5, 'user_based')
            self.get_recommendations(example_user, 5, 'item_based')
        print("\n🎉 分析完成！")
        print("📁 生成的文件:")
        print("   - cluster_analysis.png (聚类分析图)")
        print("   - model_comparison.png (模型对比图)")
        print("   - genre_preferences.png (偏好分析图)")

def main():
    analyzer = MoviePreferenceAnalyzer()
    analyzer.run_full_analysis()
    print("\n" + "="*50)
    print("示例：为用户推荐电影")
    print("="*50)
    if hasattr(analyzer, 'user_features') and len(analyzer.user_features) > 0:
        example_user = analyzer.user_features['user_id'].iloc[0]
        print(f"\n🎯 为用户 {example_user} 推荐电影:")
        print("\n📋 基于用户的协同过滤推荐:")
        analyzer.get_recommendations(example_user, 5, 'user_based')
        print("\n📋 基于物品的协同过滤推荐:")
        analyzer.get_recommendations(example_user, 5, 'item_based')

if __name__ == "__main__":
    main()