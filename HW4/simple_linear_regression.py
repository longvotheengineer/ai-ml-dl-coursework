"""
BÀI TẬP LINEAR REGRESSION ĐỠN GIẢN
Dự đoán giá nhà dựa trên diện tích
"""

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================================
# BƯỚC 1: TẠO DỮ LIỆU
# ===========================================================================================
print("="*80)
print("CHƯƠNG TRÌNH DỰ ĐOÁN GIÁ NHÀ DỰA TRÊN DIỆN TÍCH")
print("="*80)
print()

# Tạo dữ liệu mẫu: Diện tích nhà (m²)
np.random.seed(42)
X = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])  # Diện tích (m²)
# Giá nhà (triệu VNĐ) = 2 * diện tích + 100 + nhiễu ngẫu nhiên
y = 2 * X + 100 + np.random.randn(13) * 10

print("📊 DỮ LIỆU HỌC:")
print("-" * 80)
print("Diện tích (m²)  |  Giá nhà (triệu VNĐ)")
print("-" * 80)
for i in range(len(X)):
    print(f"     {X[i]:3.0f}         |        {y[i]:6.2f}")
print()

# ===========================================================================================
# BƯỚC 2: TÍNH TOÁN CÔNG THỨC LINEAR REGRESSION
# ===========================================================================================
print("📐 CÔNG THỨC LINEAR REGRESSION:")
print("-" * 80)
print("Công thức: y = w * x + b")
print("  - y: Giá nhà (triệu VNĐ)")
print("  - x: Diện tích (m²)")
print("  - w: Hệ số góc (độ dốc của đường thẳng)")
print("  - b: Hệ số tự do (giao điểm với trục y)")
print()

# Tính toán w và b theo công thức Normal Equation
n = len(X)
x_mean = np.mean(X)
y_mean = np.mean(y)

# Công thức: w = sum((x_i - x_mean) * (y_i - y_mean)) / sum((x_i - x_mean)^2)
w = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)

# Công thức: b = y_mean - w * x_mean
b = y_mean - w * x_mean

print(f"✓ Hệ số tính được:")
print(f"  w (độ dốc) = {w:.4f}")
print(f"  b (điểm cắt) = {b:.4f}")
print()
print(f"→ Phương trình đường thẳng: y = {w:.4f} * x + {b:.4f}")
print()

# ===========================================================================================
# BƯỚC 3: DỰ ĐOÁN
# ===========================================================================================
print("🔮 DỰ ĐOÁN GIÁ NHÀ:")
print("-" * 80)

# Dự đoán trên dữ liệu training
y_pred = w * X + b

# Tính sai số
errors = y - y_pred
mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)

print("Diện tích | Giá thực tế | Giá dự đoán | Sai số")
print("-" * 80)
for i in range(len(X)):
    print(f"  {X[i]:3.0f} m²  |  {y[i]:7.2f}    |   {y_pred[i]:7.2f}   | {errors[i]:+6.2f}")
print()
print(f"📊 Sai số trung bình bình phương (MSE): {mse:.2f}")
print(f"📊 Sai số trung bình (RMSE): {rmse:.2f} triệu VNĐ")
print()

# ===========================================================================================
# BƯỚC 4: DỰ ĐOÁN CHO NHÀ MỚI
# ===========================================================================================
print("🏠 DỰ ĐOÁN GIÁ CHO NHÀ MỚI:")
print("-" * 80)

# Ví dụ dự đoán cho một số diện tích mới
new_areas = [55, 85, 125, 160]
for area in new_areas:
    predicted_price = w * area + b
    print(f"Nhà có diện tích {area} m² → Giá dự đoán: {predicted_price:.2f} triệu VNĐ")
print()

# ===========================================================================================
# BƯỚC 5: VẼ ĐỒ THỊ
# ===========================================================================================
print("📈 ĐANG VẼ ĐỒ THỊ...")
print("="*80)

plt.figure(figsize=(10, 6))

# Vẽ các điểm dữ liệu thực tế
plt.scatter(X, y, color='red', s=100, alpha=0.6, label='Dữ liệu thực tế')

# Vẽ đường thẳng dự đoán
x_line = np.linspace(20, 160, 100)
y_line = w * x_line + b
plt.plot(x_line, y_line, color='blue', linewidth=2, label=f'Đường dự đoán: y = {w:.2f}x + {b:.2f}')

# Vẽ các điểm dự đoán
plt.scatter(X, y_pred, color='blue', s=50, alpha=0.8, marker='x', label='Điểm dự đoán')

# Vẽ đường nối từ điểm thực tế đến điểm dự đoán (sai số)
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], 'g--', alpha=0.3, linewidth=1)

plt.xlabel('Diện tích nhà (m²)', fontsize=12, fontweight='bold')
plt.ylabel('Giá nhà (triệu VNĐ)', fontsize=12, fontweight='bold')
plt.title('DỰ ĐOÁN GIÁ NHÀ DỰA TRÊN DIỆN TÍCH\nSử dụng Linear Regression', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Thêm text box giải thích
textstr = f'Phương trình:\ny = {w:.2f}x + {b:.2f}\n\nRMSE = {rmse:.2f} triệu VNĐ'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('linear_regression_gia_nha.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu đồ thị vào file: linear_regression_gia_nha.png")
# plt.show()  # Đã comment để không hiển thị cửa sổ, chỉ lưu file

print()
print("="*80)
print("GIẢI THÍCH:")
print("="*80)
print("• Các chấm ĐỎ: Giá nhà thực tế trong dữ liệu")
print("• Đường XANH: Đường thẳng dự đoán (fitting line)")
print("• Các dấu X XANH: Giá nhà được dự đoán bởi mô hình")
print("• Các đường nét đứt XANH LÁ: Khoảng cách sai số giữa giá thực tế và dự đoán")
print()
print("→ Mô hình Linear Regression tìm ra đường thẳng 'fit' tốt nhất")
print("  qua các điểm dữ liệu, giảm thiểu sai số giữa dự đoán và thực tế!")
print("="*80)
