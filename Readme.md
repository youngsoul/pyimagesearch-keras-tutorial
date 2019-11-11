
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

## PyimageSearch Keras Version

- See `train_simple_nn_adrian.py`

Model results:
```text
[INFO] evaluating network...
              precision    recall  f1-score   support

        cats       0.60      0.43      0.50       258
        dogs       0.48      0.54      0.51       234
       panda       0.71      0.84      0.77       258

    accuracy                           0.60       750
   macro avg       0.60      0.60      0.59       750
weighted avg       0.60      0.60      0.60       750

```

## TensorFlow2 version

- See `train_simple_nn.py`

Model results:
```text
[INFO] evaluating network...
Accuracy: 0.5986666666666667
              precision    recall  f1-score   support

        cats       0.53      0.69      0.60       258
        dogs       0.47      0.22      0.30       234
       panda       0.71      0.85      0.78       258

    accuracy                           0.60       750
   macro avg       0.57      0.59      0.56       750
weighted avg       0.58      0.60      0.57       750
```

```text
[INFO] evaluating network...
Accuracy: 0.5933333333333334
              precision    recall  f1-score   support

        cats       0.54      0.64      0.58       258
        dogs       0.51      0.22      0.31       234
       panda       0.67      0.88      0.76       258

    accuracy                           0.59       750
   macro avg       0.57      0.58      0.55       750
weighted avg       0.57      0.59      0.56       750
```

## Convolutional SmallVGGNet

- See ./pyimagesearch/smallvggnet.py

```text
[INFO] evaluating network...
              precision    recall  f1-score   support

        cats       0.65      0.85      0.74       258
        dogs       0.74      0.41      0.52       234
       panda       0.86      0.94      0.90       258

    accuracy                           0.74       750
   macro avg       0.75      0.73      0.72       750
weighted avg       0.75      0.74      0.73       750
```

```text
[INFO] evaluating network...
              precision    recall  f1-score   support

        cats       0.64      0.72      0.68       258
        dogs       0.67      0.14      0.23       234
       panda       0.63      0.99      0.77       258

    accuracy                           0.63       750
   macro avg       0.65      0.62      0.56       750
weighted avg       0.65      0.63      0.57       750
```