## Overview

VLM is a vision-language guided video object segmentation pipeline that combines open-vocabulary detection, instance segmentation, and video object tracking to enable natural language-guided target object tracking and segmentation. The system integrates three state-of-the-art components:

​Open-Vocabulary Detector​ (OWL-ViT): Enables text-prompted target localization
​Segment Anything Model (SAM): Provides high-quality instance segmentation
​XMem: Facilitates cross-frame video object tracking

## Details

Please refer to the class `VLM` in `VLM.py` and its member functions `process_first_frame` and `process_frame` for implementation details.
