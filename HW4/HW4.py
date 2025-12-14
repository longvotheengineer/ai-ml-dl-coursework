

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
  

  
# Set style cho đồ thị
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Set random seed
np.random.seed(42)


def generate_housing_data(n_samples=200):
    """
    Tạo dữ liệu giả lập về giá nhà với nhiều features
    
    Args:
        n_samples: Số lượng mẫu dữ liệu
        
    Returns:
        DataFrame chứa tất cả dữ liệu
    """
    # Tạo features
    dien_tich = np.random.uniform(30, 300, n_samples)  # 30-300 m²
    so_phong_ngu = np.random.randint(1, 6, n_samples)  # 1-5 phòng
    so_phong_tam = np.random.randint(1, 4, n_samples)  # 1-3 phòng
    tuoi_nha = np.random.uniform(0, 50, n_samples)     # 0-50 năm
    
    # Tạo giá nhà với công thức phức tạp hơn
    # Giá = 40*diện_tích + 800*phòng_ngủ + 500*phòng_tắm - 10*tuổi_nhà + noise
    noise = np.random.normal(0, 1000, n_samples)
    gia_nha = (40 * dien_tich + 
               800 * so_phong_ngu + 
               500 * so_phong_tam - 
               10 * tuoi_nha + 
               2000 +  # Base price
               noise)
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'Dien_tich_m2': dien_tich,
        'So_phong_ngu': so_phong_ngu,
        'So_phong_tam': so_phong_tam,
        'Tuoi_nha_nam': tuoi_nha,
        'Gia_nha_trieu_VND': gia_nha
    })
    
    return df


def plot_data_analysis(df):
    """
    Vẽ đồ thị phân tích dữ liệu ban đầu
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('PHÂN TÍCH DỮ LIỆU BAN ĐẦU', fontsize=16, fontweight='bold')
    
    # Distribution của từng feature
    features = ['Dien_tich_m2', 'So_phong_ngu', 'So_phong_tam', 'Tuoi_nha_nam']
    for idx, feature in enumerate(features):
        row = idx // 2
        col = idx % 2
        axes[row, col].hist(df[feature], bins=30, alpha=0.7, color=sns.color_palette()[idx], edgecolor='black')
        axes[row, col].set_xlabel(feature, fontsize=11)
        axes[row, col].set_ylabel('Tần số', fontsize=11)
        axes[row, col].set_title(f'Phân phối {feature}', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
    
    # Distribution của target
    axes[1, 0].hist(df['Gia_nha_trieu_VND'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Giá nhà (triệu VNĐ)', fontsize=11)
    axes[1, 0].set_ylabel('Tần số', fontsize=11)
    axes[1, 0].set_title('Phân phối Giá Nhà', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    axes[1, 1].axis('off')
    ax_corr = fig.add_subplot(2, 3, 6)
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax_corr, cbar_kws={'shrink': 0.8})
    ax_corr.set_title('Ma trận Correlation', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_results(X_test, y_test, y_pred, model, feature_names):
    """
    Vẽ đồ thị kết quả dự đoán
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Predicted vs Actual
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred, alpha=0.6, s=50, color='blue', edgecolor='black', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Giá thực tế (triệu VNĐ)', fontsize=11)
    ax1.set_ylabel('Giá dự đoán (triệu VNĐ)', fontsize=11)
    ax1.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = plt.subplot(2, 3, 2)
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50, color='green', edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Giá dự đoán (triệu VNĐ)', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of Residuals
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals', fontsize=11)
    ax3.set_ylabel('Tần số', fontsize=11)
    ax3.set_title('Phân phối Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Feature Importance (Coefficients)
    ax4 = plt.subplot(2, 3, 4)
    coefficients = pd.Series(model.coef_, index=feature_names).sort_values()
    colors = ['red' if x < 0 else 'green' for x in coefficients]
    coefficients.plot(kind='barh', ax=ax4, color=colors, edgecolor='black')
    ax4.set_xlabel('Coefficient Value', fontsize=11)
    ax4.set_title('Feature Importance (Coefficients)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Error Distribution
    ax5 = plt.subplot(2, 3, 5)
    errors = np.abs(residuals)
    ax5.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax5.set_xlabel('Absolute Error (triệu VNĐ)', fontsize=11)
    ax5.set_ylabel('Tần số', fontsize=11)
    ax5.set_title('Phân phối Absolute Error', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Q-Q Plot
    ax6 = plt.subplot(2, 3, 6)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot (Residuals)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_models(X_train, X_test, y_train, y_test):
    """
    So sánh nhiều models khác nhau
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1.0)': Lasso(alpha=1.0)
    }
    
    results = []
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R²': train_r2,
            'Test R²': test_r2
        })
    
    return pd.DataFrame(results)


def main():
    """
    Hàm main để chạy toàn bộ chương trình
    """
    print("="*90)
    print(" "*15 + "LINEAR REGRESSION - DỰ ĐOÁN GIÁ NHÀ SỬ DỤNG SKLEARN")
    print("="*90)
    print()
    
    # 1. Tạo và phân tích dữ liệu
    print("📊 BƯỚC 1: Tạo và Phân Tích Dữ Liệu")
    print("-" * 90)
    df = generate_housing_data(n_samples=200)
    print(f"✓ Đã tạo {len(df)} mẫu dữ liệu")
    print(f"\n📋 Thống kê mô tả:")
    print(df.describe().round(2))
    print(f"\n🔍 Thông tin dữ liệu:")
    print(df.info())
    print()
    
    # Visualize dữ liệu ban đầu
    print("📊 Đang vẽ đồ thị phân tích dữ liệu...")
    plot_data_analysis(df)
    
    # 2. Chuẩn bị dữ liệu
    print("\n🔧 BƯỚC 2: Chuẩn Bị Dữ Liệu")
    print("-" * 90)
    
    # Tách features và target
    X = df.drop('Gia_nha_trieu_VND', axis=1)
    y = df['Gia_nha_trieu_VND']
    feature_names = X.columns.tolist()
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {len(X_train)} mẫu ({len(X_train)/len(df)*100:.1f}%)")
    print(f"✓ Test set: {len(X_test)} mẫu ({len(X_test)/len(df)*100:.1f}%)")
    print(f"✓ Số features: {X.shape[1]}")
    print(f"✓ Features: {', '.join(feature_names)}")
    print()
    
    # 3. Huấn luyện model
    print("🤖 BƯỚC 3: Huấn Luyện Linear Regression Model (Sklearn)")
    print("-" * 90)
    
    # Tạo pipeline với scaling
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Train model
    model.fit(X_train, y_train)
    print("✓ Model đã được huấn luyện thành công!")
    
    # Lấy coefficients
    regressor = model.named_steps['regressor']
    print(f"\n� Model Parameters:")
    print(f"  - Intercept (b): {regressor.intercept_:.4f}")
    print(f"  - Coefficients (w):")
    for fname, coef in zip(feature_names, regressor.coef_):
        print(f"      {fname:20s}: {coef:10.4f}")
    print()
    
    # 4. Đánh giá trên Training Set
    print("📈 BƯỚC 4: Đánh Giá Model trên Training Set")
    print("-" * 90)
    y_train_pred = model.predict(X_train)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"  ✓ MSE (Mean Squared Error): {train_mse:,.2f}")
    print(f"  ✓ RMSE (Root MSE): {train_rmse:,.2f}")
    print(f"  ✓ MAE (Mean Absolute Error): {train_mae:,.2f}")
    print(f"  ✓ R² Score: {train_r2:.4f} ({train_r2*100:.2f}% variance explained)")
    print()
    
    # 5. Đánh giá trên Test Set
    print("🎯 BƯỚC 5: Đánh Giá Model trên Test Set")
    print("-" * 90)
    y_test_pred = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  ✓ MSE (Mean Squared Error): {test_mse:,.2f}")
    print(f"  ✓ RMSE (Root MSE): {test_rmse:,.2f}")
    print(f"  ✓ MAE (Mean Absolute Error): {test_mae:,.2f}")
    print(f"  ✓ R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
    print()
    
    # 6. Cross-validation
    print("🔄 BƯỚC 6: Cross-Validation (5-fold)")
    print("-" * 90)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  ✓ CV R² Scores: {cv_scores}")
    print(f"  ✓ Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print()
    
    # 7. So sánh các models
    print("⚖️  BƯỚC 7: So Sánh Các Models Khác Nhau")
    print("-" * 90)
    comparison_df = compare_models(X_train, X_test, y_train, y_test)
    print(comparison_df.to_string(index=False))
    print()
    
    # 8. Dự đoán mẫu
    print("� BƯỚC 8: Thử Nghiệm Dự Đoán")
    print("-" * 90)
    
    # Tạo các mẫu test
    test_samples = pd.DataFrame({
        'Dien_tich_m2': [50, 100, 150, 200],
        'So_phong_ngu': [2, 3, 4, 5],
        'So_phong_tam': [1, 2, 2, 3],
        'Tuoi_nha_nam': [5, 10, 20, 0]
    })
    
    predictions = model.predict(test_samples)
    
    print(f"{'DT(m²)':<10} {'Phòng ngủ':<12} {'Phòng tắm':<12} {'Tuổi(năm)':<12} {'Giá dự đoán (triệu VNĐ)':<25}")
    print("-" * 90)
    for idx, row in test_samples.iterrows():
        print(f"{row['Dien_tich_m2']:<10.0f} {row['So_phong_ngu']:<12.0f} "
              f"{row['So_phong_tam']:<12.0f} {row['Tuoi_nha_nam']:<12.0f} "
              f"{predictions[idx]:<25,.2f}")
    print()
    
    # 9. Phân tích lỗi
    print("� BƯỚC 9: Phân Tích Lỗi")
    print("-" * 90)
    residuals = y_test - y_test_pred
    print(f"  ✓ Mean Residual: {residuals.mean():.2f}")
    print(f"  ✓ Std Residual: {residuals.std():.2f}")
    print(f"  ✓ Min Residual: {residuals.min():.2f}")
    print(f"  ✓ Max Residual: {residuals.max():.2f}")
    
    # Tìm predictions tốt nhất và tệ nhất
    abs_errors = np.abs(residuals)
    best_idx = abs_errors.idxmin()
    worst_idx = abs_errors.idxmax()
    
    print(f"\n  🌟 Dự đoán tốt nhất:")
    print(f"     True: {y_test.loc[best_idx]:.2f}, Predicted: {y_test_pred[y_test.index.get_loc(best_idx)]:.2f}, Error: {abs_errors.loc[best_idx]:.2f}")
    print(f"  ⚠️  Dự đoán tệ nhất:")
    print(f"     True: {y_test.loc[worst_idx]:.2f}, Predicted: {y_test_pred[y_test.index.get_loc(worst_idx)]:.2f}, Error: {abs_errors.loc[worst_idx]:.2f}")
    print()
    
    # 10. Visualize kết quả
    print("📊 BƯỚC 10: Visualize Kết Quả Dự Đoán")
    print("-" * 90)
    print("Đang vẽ đồ thị...")
    plot_results(X_test, y_test, y_test_pred, regressor, feature_names)
    
    # Kết luận
    print()
    print("="*90)
    print(" "*30 + "HOÀN THÀNH!")
    print("="*90)
    print()
    print("💡 KẾT LUẬN:")
    print(f"   ✓ Model Linear Regression với {X.shape[1]} features")
    print(f"   ✓ R² Score trên test set: {test_r2:.4f} - Model giải thích được {test_r2*100:.2f}% variance")
    print(f"   ✓ RMSE: {test_rmse:,.2f} triệu VNĐ - Sai số trung bình")
    print(f"   ✓ Feature quan trọng nhất: {feature_names[np.argmax(np.abs(regressor.coef_))]}")
    print("   ✓ Sử dụng thư viện Sklearn giúp code ngắn gọn, dễ maintain và có nhiều tính năng")
    print("="*90)


if __name__ == "__main__":
    main()
