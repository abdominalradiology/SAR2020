# SAR 2020 - AI Masters Class

Notebooks on Intro to Python, DICOM Basics, Classification, and Segmentation for AI Masters Class
- [Lesson 01 - Intro to Python](https://github.com/twloehfelm/SAR2020/blob/master/01%20-%20Intro_to_Jupyter.ipynb)
- [Lesson 02 - DICOM Basics](https://github.com/twloehfelm/SAR2020/blob/master/02%20-%20DICOM_Basics.ipynb)
- [Lesson 03 - Classification](https://github.com/twloehfelm/SAR2020/blob/master/03%20-%20Image_Classifier.ipynb)
- [Lesson 04 - Segmentation](https://github.com/twloehfelm/SAR2020/blob/master/04%20-%20Segmentation.ipynb)

## Lesson 01 - Intro to Python
Very basic introduction to Python and the Jupyter/Colab environment. If you've never visited the command line, start here.

## Lesson 02 - DICOM Basics
Brief introduction to reading DICOM files including information from the DICOM header and the pixel array.
Relies on pydicom and concepts from fastaiv2 medical imaging libraries.

## Lesson 03 - Classification
Build a simple image classifier to identify a chest x-ray as either frontal or lateral projection.
Based on fastai library and Deep Learning for Coders lesson.

## Lesson 04 - Segmentation
Adapts Facebook Research's detectron2 library to read native DICOM. We'll feed single-channel full bit-depth Hounsfield unit pixel data into detectron2 - DICOM and segmentations are from the recent [Combined Healthy Abdominal Organ Segmentation (CHAOS) grand challenge](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/).
 
Key modifications to detectron2 to enable reading DICOM natively:
- Custom data mapper to read the dcm file pixel_array and convert to Houndfield units.
   - Applies random brightness and random contrast run time augmentations.
   - Does NOT apply random flipping as horizontal flipping seems to lead to some liver-spleen confusion for the algorithm.
   - Sets mask type to "bitmask" from default "polygons"
- Custom get_liver_dicts referenced by DatasetCatalog.register function
  - Loads from _train, _val, or _test file lists
  - Saves path to .dcm file as dataset_dict["file_name"]
    - Default detectron data mapper simply opens the image file from file_name. Our custom data mapper will instead read the dcm pixel array, convert to Hounsfield units, and load the image array from that
  - Gets image height and width from dcm.pixel_array
  - Creates bounding box automatically from the given mask (via separate bbox function)
  - Converts binary PNG mask to RLE format using pycocotools library
- Modifies base configuration file for the trainer
  - cfg.SOLVER.IMS_PER_BATCH, BASE_LR, and MAX_ITER are good places to start tweaking to modify performance
  - cfg.INPUT.FORMAT = "F" tells the model to expect single channel float array instead of default "RGB"
  - cfg.INPUT.MASK_FORMAT = "bitmask" change from default "polygons"
- Extends the DefaultTrainer as LiverTrainer
  - To use custom data mapper in the build_train_loader and build_test_loader functions.
  - To use custom config file with modifications as above
- Replaces the DefaultPredictor class with LiverPredictor
  - Change torch.as_tensor to use single channel input

- Note that windowing is only done when displaying example images or overlaid inference results
  - Pixel data that are loaded into the network are NOT windowed since that would needlessly compress the dynamic range to the (presumed) detriment of the model's learning.
    - In this way windowing in python/ML applications means something slightly different than windowing at a diagnostic workstation. Radiolgists know but may not fully appreciate (and do not seem to communicate to computer scientists) that windowing at the workstation doesn't change the actual pixel values - an ROI will be the same regardless of the window/level setting applied since it is measuring the true HU, not the screen's pixel greyscale values. Windowing in python/ML, on the other hand, fundamentally changes the underlying pixel values.
