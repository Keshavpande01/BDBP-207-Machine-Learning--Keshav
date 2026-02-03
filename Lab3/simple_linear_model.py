import numpy as np

class LinearRegression :
    def __init__ (self, learning_rate=0.01,epochs=100,):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weight=0
        self.bias=0

    def predict(self,X):
        return self.weight * X + self.bias

    def compute_loss(self,y_pred,y):
        return np.mean(( y - y_pred)**2)

    def fit(self,X,y):
        n=len(X)

        for i in range(self.epochs):
            y_pred=self.predict(X)
            dw=(-2/n) * np.mean(X * (y - y_pred))
            db=(-2/n) * np.mean(y - y_pred)
            self.weight -=self.learning_rate *dw
            self.bias -=self.learning_rate * db

            if i % 100 ==0:
                loss=self.compute_loss(y_pred,y)
                print(f"Iteration: {i},weight={self.weight:.4f},Bias={self.bias:.4f}")


    def evaluate(self,X,y):
        y_pred=self.predict(X)
        loss=self.compute_loss(y_pred,y)
        print(f"Final loss : {loss:.4f}")


X=np.array ([1,2,3,4,5])
y=np.array([2,4,6,8,10])

model=LinearRegression(learning_rate=0.01,epochs=10000)

model.fit(X,y)

model.evaluate(X,y)

print("predicted price for size 6 :",model.predict(6))