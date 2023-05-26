from knn import KNNClassifier

if __name__ == '__main__':
    knn = KNNClassifier()
    knn.load_dataset()
    knn.split_dataset()
    knn.train_model()
    knn.evaluate_model()
    knn.plot_decision_boundary()
