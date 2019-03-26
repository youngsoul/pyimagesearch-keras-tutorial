
# PyImageSearch Keras Tutorial

[PyImageSearchBlog](https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)

This repo contains my work in reading through the above blog.


When running *train_simple_nn.py* the best the model could do was:

```text
Accuracy: 0.6413333333333333
              precision    recall  f1-score   support

        cats       0.58      0.43      0.50       236
        dogs       0.50      0.63      0.55       236
       panda       0.84      0.83      0.83       278

   micro avg       0.64      0.64      0.64       750
   macro avg       0.64      0.63      0.63       750
weighted avg       0.65      0.64      0.64       750

```

for a model constructed as:
```python
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))


model.add(Dense(len(lb.classes_), activation="softmax"))

```