Intended use: 
- connect Google Colab to the GitHub repo so the latest version of the script is running
- take short ~15s video clips using a USB camera on guvcview at the camera resolution you need (preferably 30fps)
- follow the embedded instructions for using the script properly and exporting the dataset


## Fixed-size output and optional centering

Cell 2 saves 640×480 image/label pairs from different input resolutions. Set:

```python
CENTER_ON_KEYPOINTS = False
```

- `False` uses the image center as the crop center. This is the default.
- `True` uses the bounding-box center of the currently visible keypoints. The crop follows the pick and is clamped to the source image edges.

Both modes crop original pixels and add black padding when a dimension is smaller than the output. The saved images preserve source pixel scale, including changes in apparent pick size with distance. Keypoints receive the same translation as the pixels, then YOLO coordinates are normalized by 640×480. Tracking may use a smaller temporary frame internally; exported images use the original decoded pixels.

Select points on the original input frame. Points outside the saved crop become invisible. Cell 5 skips frames with fewer than `MIN_VISIBLE` points, and Cell 6 removes frames with fewer than all five points. Inspect these results, particularly for picks larger than the crop. The two centering modes use separate output folders. Choose one mode per dataset, rerun Cell 2 after changing it, and export that mode's ZIP before ending the Colab session.

For training a detector over the camera view, include varied target positions, orientations and distances. Keypoint-centered cropping is optional so you can choose the geometry appropriate to the dataset.

CPU geometry and synthetic-export checks can be run in a Python environment with NumPy, OpenCV, and PyTorch:

```bash
python -m unittest discover -s tests -v
```
