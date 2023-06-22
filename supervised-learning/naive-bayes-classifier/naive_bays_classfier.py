import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# データ生成
np.random.seed(0)

# クラス1のデータ生成（平均: [2, 2]、共分散行列: [[1, 0], [0, 1]]）
class1_data = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 100)
class1_labels = np.zeros(100)

# クラス2のデータ生成（平均: [0, 0]、共分散行列: [[1, 2], [3, 1]]）
class2_data = np.random.multivariate_normal([0, 0], [[1, 2], [3, 1]], 100)
class2_labels = np.ones(100)

# データ結合
X = np.vstack((class1_data, class2_data))
y = np.concatenate((class1_labels, class2_labels))

# ナイーブベイズ分類器の学習と予測
classifier = GaussianNB()
classifier.fit(X, y)
predictions = classifier.predict(X)

# パフォーマンスメトリクスの計算
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

# 結果の出力
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# データと決定境界の可視化
plt.figure(figsize=(8, 6))
plt.scatter(class1_data[:, 0], class1_data[:, 1], c='blue', label='Class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1], c='red', label='Class 2')

# メッシュグリッドの作成
h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# メッシュグリッド上の各点での予測結果を計算
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 決定境界をプロット
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.colorbar()

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Naive Bayes Classifier')
plt.legend()
plt.show()