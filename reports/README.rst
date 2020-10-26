Results
=========================================

Semi-Supervised
----------------

**Pipeline**: 

Doc2Vec (trained on APNews dataset)

    \|

UMAP reduction from 300 -> 256 dimensions
    \|

ivis_ (siamese network) 256 -> 1 dimension (outlier score)
    \|

use interquartile range to get outliers (fixed contamination)

|


**Dataset**: RVL-CDIP

**Train-test split**: 80 - 20

**Labeled data**: 0.8 or 1.0 (all)

**K-fold**: k = 4

**Training results**:

.. image:: semisupervised/pairwise-training.png

**Test results**:

.. image:: semisupervised/pairwise-test.png

**Difference 0.8 and 1.0 of data labelled**:

.. image:: semisupervised/pairwise-08-10-diff.png


**Split Data into inliers and outliers**:

**Inlier data**: letter, form, email, invoice

**Outlier data**: handwritten, advertisement, scientific report, scientific publication, 
specification, file folder, news article, budget, presentation, questionnaire, 
resume, memo

**Table** - Contamination and Labeled data:

===========  ==========  =======  =========  ========  ========  ==========  =========
..             f1_macro    in_f1    in_prec    in_rec    out_f1    out_prec    out_rec
===========  ==========  =======  =========  ========  ========  ==========  =========
(0.05, 0.1)        0.46     0.89       0.95      0.85      0.03        0.02       0.05
(0.05, 0.4)        0.45     0.9        0.95      0.85      0.01        0.01       0.02
(0.05, 0.7)        0.46     0.9        0.95      0.86      0.01        0.01       0.02
(0.05, 1.0)        0.51     0.97       0.95      0.99      0.04        0.34       0.02
(0.1, 0.1)         0.43     0.85       0.89      0.81      0.02        0.01       0.03
(0.1, 0.4)         0.45     0.87       0.9       0.85      0.02        0.01       0.02
(0.1, 0.7)         0.43     0.85       0.89      0.81      0.02        0.01       0.03
(0.1, 1.0)         0.54     0.95       0.91      0.99      0.13        0.64       0.08
(0.2, 0.1)         0.43     0.78       0.81      0.75      0.09        0.08       0.11
(0.2, 0.4)         0.43     0.78       0.81      0.76      0.08        0.06       0.09
(0.2, 0.7)         0.43     0.78       0.81      0.76      0.07        0.06       0.08
(0.2, 1.0)     **0.67**     0.92       0.87      0.98  **0.41**        0.76       0.28
===========  ==========  =======  =========  ========  ========  ==========  =========


Supervised
-----------

**Pipeline**: 

Doc2Vec (trained on APNews dataset)

    \|

UMAP reduction from 300 -> 256 dimensions
    \|

Small FC-NN
    \|

256 - 128 - 64 - 16 - 1

|


**Dataset**: RVL-CDIP

**Inlier data**: letter, form, email, invoice

**Outlier data**: handwritten, advertisement, scientific report, scientific publication, 
specification, file folder, news article, budget, presentation, questionnaire, 
resume, memo


**Contamination**: 0.1

**Training data**: 

====  =====
  0      1
====  =====
 918   9191
====  =====



**Test data input**:

===  ====
  0     1
===  ====
230  2298
===  ====

**Outputs**:


.. image:: semisupervised/distribution_output_supervised.png
   :width: 600

**Predictions** (with threshold 0.5):

===  ====
  0     1
===  ====
 96  2432
===  ====


**Scores**:

====  ==========  =======  ========  =========  ========  =========  ==========
  ..    f1_macro    in_f1    in_rec    in_prec    out_f1    out_rec    out_prec
====  ==========  =======  ========  =========  ========  =========  ==========
   0       0.733    0.965     0.991       0.94     0.501      0.365         0.8
====  ==========  =======  ========  =========  ========  =========  ==========


.. _ivis: https://github.com/beringresearch/ivis