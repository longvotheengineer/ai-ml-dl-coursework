"""
KNN-based Digit Recognition System using scikit-learn

Problem Statement:
Implement a K-Nearest Neighbors (KNN) classifier to recognize handwritten digits from the MNIST dataset.
The goal is to correctly classify digits from 0 to 9 based on their pixel values.

Tasks:
1. Load and preprocess the MNIST dataset (28x28 pixel images)
2. Train KNN models with different K values
3. Evaluate the models' performance
4. Visualize examples and their classifications
5. Analyze the results

Author: thuong_trandinh
Date: October 21, 2025
"""



import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def read_idx_images(filepath):
    """
    Read IDX file format images
    
    Parameters:
        filepath (str): Path to IDX file
    
    Returns:
        numpy.ndarray: Images
    """
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images

def read_idx_labels(filepath):
    """
    Read IDX file format labels
    
    Parameters:
        filepath (str): Path to IDX file
    
    Returns:
        numpy.ndarray: Labels
    """
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_data():
    """
    Load and preprocess the MNIST dataset from local files
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, images_train, images_test
    """
    print("Loading MNIST dataset from local files...")
    
    # Path to the MNIST dataset files
    mnist_dir = '/home/thuong/DAIHOC/HK_251/AI/project/archive'
    
    # Load training data
    train_images_path = os.path.join(mnist_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(mnist_dir, 'train-labels.idx1-ubyte')
    
    # Load test data
    test_images_path = os.path.join(mnist_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(mnist_dir, 't10k-labels.idx1-ubyte')
    
    # Read data
    X_train_full = read_idx_images(train_images_path).astype('float32')
    y_train_full = read_idx_labels(train_labels_path)
    
    X_test_full = read_idx_images(test_images_path).astype('float32')
    y_test_full = read_idx_labels(test_labels_path)
    
    print(f"Original MNIST: {X_train_full.shape[0]} training images, {X_test_full.shape[0]} test images")
    
    # Normalize pixel values to [0, 1] range
    X_train_full = X_train_full / 255.0
    X_test_full = X_test_full / 255.0
    
    # Use a subset to make computation faster with KNN
    # Using 10,000 samples for training and 2,000 for testing
    n_train_samples = 10000
    n_test_samples = 2000
    
    if X_train_full.shape[0] > n_train_samples:
        # Ensure balanced classes by stratified sampling
        train_indices = []
        for digit in range(10):
            digit_indices = np.where(y_train_full == digit)[0]
            # Take approximately n_train_samples/10 samples per digit
            samples_per_digit = n_train_samples // 10
            selected = np.random.choice(digit_indices, samples_per_digit, replace=False)
            train_indices.extend(selected)
        
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
        print(f"Using {len(X_train)} samples from MNIST training set")
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    if X_test_full.shape[0] > n_test_samples:
        # Similar stratified sampling for test set
        test_indices = []
        for digit in range(10):
            digit_indices = np.where(y_test_full == digit)[0]
            samples_per_digit = n_test_samples // 10
            selected = np.random.choice(digit_indices, samples_per_digit, replace=False)
            test_indices.extend(selected)
        
        X_test = X_test_full[test_indices]
        y_test = y_test_full[test_indices]
        print(f"Using {len(X_test)} samples from MNIST test set")
    else:
        X_test = X_test_full
        y_test = y_test_full
    
    # Reshape for visualization
    images_train = X_train.reshape(-1, 28, 28)
    images_test = X_test.reshape(-1, 28, 28)
    
    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Data loaded and preprocessed: {X_train.shape[0]} training examples, {X_test.shape[0]} test examples")
    print(f"Each digit is represented as a 28x28 image with {X_train.shape[1]} features after flattening")
    
    return X_train, X_test, y_train, y_test, images_train, images_test

def evaluate_model(k_values, X_train, X_test, y_train, y_test):
    """
    Evaluate KNN models with different k values using scikit-learn
    
    Parameters:
        k_values (list): List of k values to evaluate
        X_train, X_test, y_train, y_test: Training and testing data
        
    Returns:
        dict: Dictionary with k values as keys and accuracies as values
    """
    import os
    accuracies = {}
    training_times = {}
    prediction_times = {}
    
    # Đường dẫn đến file lưu trữ kết quả mô hình
    cache_dir = 'model_results'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Kiểm tra kích thước dữ liệu để xác định prefix cho file đầu ra
    img_size = int(np.sqrt(X_train.shape[1]))
    dataset_prefix = f"mnist_{img_size}x{img_size}"
    
    for k in k_values:
        result_file = os.path.join(cache_dir, f"{dataset_prefix}_k{k}_results.npz")
        
        # Kiểm tra xem đã có kết quả từ trước chưa
        if os.path.exists(result_file):
            print(f"\nTải kết quả đã lưu cho k={k}...")
            data = np.load(result_file, allow_pickle=True)
            accuracy = data['accuracy'].item()
            train_time = data['train_time'].item()
            pred_time = data['pred_time'].item()
            cm = data['confusion_matrix']
            report = data['report'].item()
            
            accuracies[k] = accuracy
            training_times[k] = train_time
            prediction_times[k] = pred_time
            
            print(f"KNN với k={k}:")
            print(f"  Độ chính xác = {accuracy:.4f}")
            print(f"  Thời gian huấn luyện: {train_time:.4f} giây")
            print(f"  Thời gian dự đoán: {pred_time:.4f} giây")
            
            print("\nBáo cáo phân loại:")
            print(report)
            
        else:
            print(f"\nHuấn luyện KNN với k={k}...")
            
            # Khởi tạo mô hình với tối ưu hóa
            knn = KNeighborsClassifier(
                n_neighbors=k, 
                n_jobs=-1,  # Sử dụng tất cả CPU có sẵn
                algorithm='auto',  # Tự động chọn thuật toán tìm kiếm tốt nhất
                leaf_size=30  # Tham số tối ưu cho cấu trúc dữ liệu cây
            )
            
            # Huấn luyện mô hình và đo thời gian
            start_time = time.time()
            knn.fit(X_train, y_train)
            train_end_time = time.time()
            training_times[k] = train_end_time - start_time
            
            # Dự đoán trên tập kiểm tra và đo thời gian
            print(f"Đang dự đoán với {len(X_test)} mẫu kiểm tra...")
            start_time = time.time()
            y_pred = knn.predict(X_test)
            pred_end_time = time.time()
            prediction_times[k] = pred_end_time - start_time
            
            # Tính độ chính xác
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[k] = accuracy
            
            # Tạo báo cáo phân loại
            report = classification_report(y_test, y_pred)
            
            # Tính ma trận nhầm lẫn
            cm = confusion_matrix(y_test, y_pred)
            
            # Lưu kết quả
            np.savez(
                result_file,
                accuracy=accuracy,
                train_time=training_times[k],
                pred_time=prediction_times[k],
                confusion_matrix=cm,
                report=report
            )
            
            # Hiển thị kết quả
            print(f"KNN với k={k}:")
            print(f"  Độ chính xác = {accuracy:.4f}")
            print(f"  Thời gian huấn luyện: {training_times[k]:.4f} giây")
            print(f"  Thời gian dự đoán: {prediction_times[k]:.4f} giây")
            
            print("\nBáo cáo phân loại:")
            print(report)
        
        # Vẽ ma trận nhầm lẫn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title(f'Ma trận nhầm lẫn ({dataset_prefix}, k={k})')
        
        # Lưu ma trận nhầm lẫn
        confusion_matrix_file = f'{dataset_prefix}_confusion_matrix_k{k}.png'
        plt.savefig(confusion_matrix_file)
        print(f"Đã lưu ma trận nhầm lẫn cho k={k} vào '{confusion_matrix_file}'")
        plt.close()
    
    return accuracies, training_times, prediction_times

def visualize_predictions(images_test, y_test, y_pred, num_samples=10):
    """
    Visualize some test examples and their predictions
    
    Parameters:
        images_test (numpy.ndarray): Original test images (not standardized)
        y_test (numpy.ndarray): True labels
        y_pred (list): Predicted labels
        num_samples (int): Number of samples to visualize
    """
    # Choose random samples
    indices = np.random.choice(len(y_test), num_samples, replace=False)
    
    # Plot samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Get the image (already in 28x28 shape from our load_data function)
        img = images_test[idx]
        
        # Plot image
        axes[i].imshow(img, cmap='gray')
        
        # Set title with true and predicted labels
        color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        axes[i].set_title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}', color=color)
        
        # Remove ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    print("Saved predictions visualization to 'mnist_predictions.png'")
    plt.close()

def plot_model_metrics(k_values, accuracies, training_times, prediction_times):
    """
    Plot metrics for different k values
    
    Parameters:
        k_values (list): List of k values
        accuracies (dict): Dictionary with k values as keys and accuracies as values
        training_times (dict): Dictionary with k values as keys and training times as values
        prediction_times (dict): Dictionary with k values as keys and prediction times as values
    """
    # Convert dictionaries to lists
    k_list = list(accuracies.keys())
    acc_list = list(accuracies.values())
    train_times = [training_times[k] for k in k_list]
    pred_times = [prediction_times[k] for k in k_list]
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy vs k
    ax1.plot(k_list, acc_list, marker='o', linestyle='-', color='blue')
    ax1.set_xlabel('k value')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('MNIST: Accuracy vs. k value for KNN (28x28)')
    ax1.grid(True)
    
    # Plot times vs k
    ax2.plot(k_list, train_times, marker='s', linestyle='-', label='Training Time', color='green')
    ax2.plot(k_list, pred_times, marker='^', linestyle='-', label='Prediction Time', color='red')
    ax2.set_xlabel('k value')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('MNIST: Training and Prediction Times (28x28)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('mnist_knn_metrics.png')
    print("Saved KNN metrics to 'mnist_knn_metrics.png'")
    plt.close()

def main():
    """Main function to run the program"""
    print("Bắt đầu nhận dạng chữ số sử dụng KNN với scikit-learn")
    print("-" * 60)
    print("Chương trình này sẽ:")
    print("1. Tải tập dữ liệu MNIST (hoặc digits nếu MNIST không khả dụng)")
    print("2. Huấn luyện mô hình KNN với các giá trị k khác nhau")
    print("3. Đánh giá hiệu suất và lưu kết quả")
    print("-" * 60)
    
    # Đặt seed ngẫu nhiên cho khả năng tái tạo kết quả
    np.random.seed(42)
    
    # Tải dữ liệu
    print("\nĐang tải dữ liệu...")
    start_time = time.time()
    X_train, X_test, y_train, y_test, images_train, images_test = load_data()
    load_time = time.time() - start_time
    print(f"Đã tải dữ liệu trong {load_time:.2f} giây")
    
    # Display some sample images
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.suptitle('Sample MNIST Digits from Training Data (28x28 pixels)')
    plt.savefig('mnist_samples.png')
    print("Saved sample digits to 'mnist_samples.png'")
    plt.close()
    
    # Evaluate model with different k values
    print("\nEvaluating model with different k values...")
    k_values = [1, 3, 5, 7, 9, 11]
    accuracies, training_times, prediction_times = evaluate_model(k_values, X_train, X_test, y_train, y_test)
    
    # Vẽ biểu đồ các chỉ số
    print("\nĐang vẽ biểu đồ các chỉ số...")
    plot_model_metrics(k_values, accuracies, training_times, prediction_times)
    
    # Xác định kích thước ảnh để đặt tên file phù hợp
    img_size = int(np.sqrt(X_train.shape[1]))
    dataset_prefix = f"mnist_{img_size}x{img_size}"
    
    # Lưu chỉ số so sánh vào file với nhãn số trên mỗi cột
    plt.figure(figsize=(10, 6))
    acc_values = [accuracies[k] for k in k_values]
    bars = plt.bar(k_values, acc_values, color='skyblue')
    
    # Thêm giá trị số trên đỉnh mỗi cột
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc_values[i]:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Giá trị k')
    plt.ylabel('Độ chính xác')
    plt.title(f'KNN - Độ chính xác theo giá trị k ({img_size}x{img_size} pixel)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(acc_values) + 0.05)  # Điều chỉnh khoảng trục y để chừa chỗ cho nhãn
    accuracy_file = f'{dataset_prefix}_accuracy_comparison.png'
    plt.savefig(accuracy_file)
    print(f"Đã lưu biểu đồ so sánh độ chính xác vào '{accuracy_file}'")
    plt.close()
    
    # Tìm giá trị k tốt nhất
    best_k = max(accuracies, key=accuracies.get)
    print(f"\nGiá trị k tốt nhất: {best_k} với độ chính xác: {accuracies[best_k]:.4f}")
    
    # Huấn luyện mô hình với giá trị k tốt nhất
    print(f"\nĐang huấn luyện mô hình cuối cùng với k={best_k}...")
    model_file = f'model_results/{dataset_prefix}_best_model_k{best_k}.pkl'
    import os
    import pickle
    
    if os.path.exists(model_file):
        print(f"Tải mô hình đã lưu từ '{model_file}'...")
        with open(model_file, 'rb') as f:
            final_model = pickle.load(f)
        y_pred = final_model.predict(X_test)
    else:
        print("Huấn luyện mô hình mới...")
        # Cấu hình tối ưu cho mô hình cuối cùng
        final_model = KNeighborsClassifier(
            n_neighbors=best_k,
            weights='uniform',  # có thể thử 'distance' cho trọng số dựa trên khoảng cách
            algorithm='auto',
            leaf_size=30,
            p=2,  # sử dụng metric Minkowski với p=2 (khoảng cách Euclidean)
            n_jobs=-1
        )
        final_model.fit(X_train, y_train)
        
        # Lưu mô hình để sử dụng sau này
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(final_model, f)
        print(f"Đã lưu mô hình vào '{model_file}'")
        
        # Dự đoán trên tập kiểm tra
        y_pred = final_model.predict(X_test)
    
    # Trực quan hóa một số dự đoán
    print("\nĐang trực quan hóa các dự đoán...")
    visualize_predictions(images_test, y_test, y_pred)
    
    # Kết luận
    print("\nKết luận:")
    print(f"Mô hình KNN trên {dataset_prefix} đạt độ chính xác {accuracies[best_k]:.4f} với k={best_k}.")
    print(f"Thời gian huấn luyện cho mô hình tốt nhất: {training_times[best_k]:.4f} giây")
    print(f"Thời gian dự đoán cho mô hình tốt nhất: {prediction_times[best_k]:.4f} giây")
    print(f"Số chiều đặc trưng: {X_train.shape[1]} ({img_size}x{img_size} pixel)")
    print(f"Số mẫu huấn luyện sử dụng: {len(X_train)}")
    print("\nƯu điểm của KNN cho nhận dạng chữ số:")
    print("1. Đơn giản, dễ hiểu và dễ triển khai")
    print("2. Không có giai đoạn huấn luyện phức tạp (chỉ lưu trữ dữ liệu)")
    print("3. Tự nhiên xử lý được bài toán đa lớp")
    print("4. Có thể đạt độ chính xác cao khi điều chỉnh tham số phù hợp")
    
    print("\nHạn chế của KNN:")
    print("1. Dự đoán có thể chậm với tập dữ liệu lớn")
    print("2. Yêu cầu chuẩn hóa đặc trưng")
    print("3. Bị ảnh hưởng bởi 'lời nguyền của chiều' (curse of dimensionality)")
    print("4. Tốn nhiều bộ nhớ (phải lưu toàn bộ tập huấn luyện)")
    
    print("\nĐể cải thiện hiệu suất trong bài toán nhận dạng chữ số, có thể xem xét:")
    print("- Support Vector Machines (SVM)")
    print("- Convolutional Neural Networks (CNN)")
    print("- Random Forests hoặc Gradient Boosted Trees")
    
    print("\nQuá trình hoàn tất!")
    print(f"Tất cả kết quả đã được lưu trong thư mục hiện tại")
    
    # In tóm tắt về các file đầu ra
    print("\nCác file kết quả:")
    print(f"1. {dataset_prefix}_accuracy_comparison.png - Biểu đồ so sánh độ chính xác với các giá trị k")
    print(f"2. mnist_knn_metrics.png - Biểu đồ thời gian và độ chính xác")
    for k in k_values:
        print(f"3. {dataset_prefix}_confusion_matrix_k{k}.png - Ma trận nhầm lẫn cho k={k}")
    print(f"4. mnist_predictions.png - Trực quan hóa các dự đoán")

if __name__ == "__main__":
    main()