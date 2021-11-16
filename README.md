## Introduction

This is the repository of Megvii Challenge in paper re-implementation.

This work is based on [IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID](https://arxiv.org/pdf/2108.02413.pdf).

## Dataset Introduction

Market1501 by Tsinghua: Inside the dataset, there are five folders -- `bounding_box_test`, `bounding_box_train`, `gt_bbox`, `gt_query` and `query`. The `bounding_box_test` is your gallery, where there is good detection and could be a match to the query and irrelevant detection. The `gt_bbox` is the manually assigned bounding box and used to test the quality of the bounding box. `gt_query ` is used to test which is a good match (same person within/across cameras) and which is a bad match (different person within camera / across camera). `query` has six images for each person and no repetition to test set.

DUKEMTMC by Duke: Inside the dataset, there are three folders -- `bounding_box_test`, `bounding_box_train`, `query`. Like a simplified version of Market1501. `query` has three images for each person. It's simpler since a few of persons are shot under only one camera and there is no irrelevant detection.
