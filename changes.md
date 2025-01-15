Added Cuda streams optimization to prediction.

To run:
```python3 test.py```
(Ensure to change to the model you are using and the dataset path)

A class `StreamOptimizedPredictor` is added to ultralytics/engine/predictor.py which has the optimizations over the previous implementation.

The file ultralytics/models/yolo/detect/predict.py is modified to use the new class. Change it back to `BasePredictor` for a benchmark.

Using a tensorrt yolovn11.engine model on my rtx3080ti laptop gpu for 5000 images, the original implementation took 30 seconds whereas the new implementation took 20 seconds.

It still has issues like:

1. The logs numbering is off by 1.
2. The profiler speeds doesn't works accurately due to asynchronous implementation and a Cuda profiler needs to be used to get that.
3. The final labels prediction value doens't match even though the labels are correct. Can be verified using a `diff dir1 dir2` command.

As this was just a proof of concept from my side i am not fixing the issues currently. Feel free to test it out to see the speedup.