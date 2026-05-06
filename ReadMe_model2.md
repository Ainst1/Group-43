- model2.ipynb

## Step 1:

The training data directory, distribution and shape have been obtained.

## Step 2:

Data is loaded into memory.

## Step 3:

Data is split for trainig and validation -> 80/20 split.

## Step 4:

CNN is built. The network has two blocks that learn features, followed by a classifier.

Block 1 — scans the image for simple patterns like edges and strokes.
Block 2 — combines those simple patterns into complex shapes like loops and curves.
Classifier — looks at everything found and decides which digit it is.

One image enters as (32, 32, 1) and exits as 10 probabilities — one per digit.
The highest probability is the prediction.

## Step 5:

Model is compiled with Adam optimizer and trained for 30 epochs with 80/20 validation split.
Best validation accuracy: 99.88% at epoch 28.

## Step 6:

Predictions generated on 3,000 test images and saved to submission.csv.

### Result : 0.9920212
