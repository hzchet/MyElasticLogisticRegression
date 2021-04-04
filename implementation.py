import batch
import numpy as np


def logit(x, w):
    return np.dot(x, w)


def sigmoid(h):
    return 1 / (1 + np.exp(-h))


class MyElasticLogisticRegression(object):
    def __init__(self, l1_coef, l2_coef):
        """

        :param l1_coef: коэффициент L1 регуляризации
        :param l2_coef: коэффициент L2 регуляризации
        """
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            # Вектор столбец в качестве весов
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)

        losses = []
        # Кладем в лист losses лосс на каждом батче

        for i in range(epochs):
            for X_batch, y_batch in batch.generate_batches(X_train, y, batch_size):
                y_pred = self._predict_proba_internal(X_batch)
                # Вычисляем loss на текущем батче
                loss = self.__loss(y_batch, y_pred)
                losses.append(loss)

                # Обновляем self.w по формуле градиентного спуска
                self.w -= lr * self.get_grad(X_batch, y_batch, y_pred)
        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        """
        Принимает на вход X_batch с уже добавленной колонкой единиц.
        Выдаёт градиент функции потерь в логистической регрессии с регуляризаторами
        как сумму градиентов функции потерь на всех объектах батча + регуляризационное слагаемое

        :returns вектор столбец градиентов для каждого из весов (np.array[n_features + 1])
        """

        grad_basic = np.dot(X_batch.T, (predictions - y_batch))

        w = self.get_weights()
        w[0] = 0
        lambdaI = self.l2_coef * np.eye(w.shape[0])
        grad_l2 = 2 * lambdaI @ w

        signw = np.sign(w)
        grad_l1 = self.l1_coef * signw

        return grad_basic + grad_l1 + grad_l2

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
