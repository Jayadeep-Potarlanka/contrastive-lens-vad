# Contrastive Lens - Video Anomaly Detection

This project develops a comprehensive framework for video anomaly detection using a variety of self-supervised learning techniques. The system is designed to learn the characteristics of normal video behavior from unlabeled data and then identify deviations from this learned norm as anomalies.

The core idea is to train models on several "pretext" tasks where the data itself provides the supervision signal. By learning to solve these tasks, the models build a rich understanding of temporal and spatial patterns in video data. Anomalies are then detected by measuring how well new, unseen video snippets conform to the patterns learned during self-supervised training.

## Key Features

-   **Multi-Task Self-Supervised Learning:** The framework combines multiple pretext tasks for robust feature learning, including:
    -   Frame Prediction & Reconstruction
    -   Contrastive Learning (MoCo V2)
    -   Transformation Classification (e.g., predicting rotation, temporal shuffling)
-   **Deep Spatio-Temporal Feature Extraction:** Utilizes a 3D ResNet-based encoder (`TACNetEncoder`) to effectively capture both spatial appearance and temporal dynamics within video snippets.
-   **Data Augmentation for Video:** Employs a suite of video-specific data augmentations to create positive and negative pairs for contrastive learning and to generate training data for classification tasks. This includes temporal reversal, frame shuffling, speed variation, and jittered rotations.
-   **Holistic Anomaly Scoring:** Anomaly detection is based on a combination of multiple "regularity scores" derived from the different self-supervised tasks to make a more reliable final decision.

## Dataset

This project utilizes the **UCSD Anomaly Detection Ped1 Dataset** to train and evaluate the model's ability to detect unusual events in pedestrian walkways. The dataset was captured using a stationary, elevated camera observing walkways with varying crowd densities. The official dataset can be found at [http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm]

The `Ped1` subset features clips with groups of people moving towards and away from the camera, which includes some perspective distortion.

## Objective

In the context of this dataset, **normal events** exclusively show pedestrians walking, while **abnormal events** are naturally occurring anomalies such as bikers, skaters, small carts, and people walking in the grass instead of on the walkway. The goal of this project is to accurately identify these abnormal events within the video footage.


## Models and Methodology

The anomaly detection pipeline consists of several interconnected components, from data preparation to final scoring.

### 1. Data Augmentation and Preprocessing

The foundation of the self-supervised approach lies in the data augmentation strategy. The `data_generator` function processes input videos by:

1.  Extracting 8-frame snippets from `.tif` files.
2.  Applying a series of augmentations to create diverse training examples:
    -   `temporal_reverse`: Reverses the frame order.
    -   `shuffle_snippet`: Randomizes the order of frames.
    -   `vary_speed`: Simulates faster or slower motion.
    -   `jittered_rotation`: Applies randomized rotations to frames.

The `preprocess_snippets_for_encoding` function then prepares these snippets for the neural networks by resizing, normalizing, and structuring them into 5D tensors `(B, C, T, H, W)`.

### 2. Core Models

#### TACNetEncoder

This is a 3D Convolutional Neural Network (CNN) with a ResNet-style architecture. Its primary role is to act as a feature extractor, taking raw video snippets and encoding them into a compact, high-dimensional latent vector that captures their essential spatio-temporal features.

#### Self-Supervised Models

The project employs multiple models trained on different pretext tasks:

*   **Frame Prediction/Reconstruction Decoders:** These are trained alongside the encoder in an autoencoder-like setup.
    *   The **Reconstruction Decoder** (`Decoder`) learns to regenerate the original 8-frame snippet from its encoded representation.
    *   The **Prediction Decoder** (`FramePredictionDecoder`) is given the first 7 frames of a snippet and tasked with predicting the final 8th frame.
    Success in these tasks indicates that the encoder is capturing information vital for understanding motion and appearance.

*   **MoCoV2 (Momentum Contrast):** This model is a key component for learning representations through contrastive learning. It's trained to differentiate between similar and dissimilar video snippets.
    *   A "query" snippet (the original) and a "key" snippet (an augmented version) are encoded.
    *   Using the `ContrastiveLoss` (InfoNCE), the model is trained to pull the representations of the query and a positive key together while pushing them apart from a dynamic queue of negative samples.
    *   The use of a momentum encoder ensures that the dictionary of negative samples is large and consistent.

*   **Self-Supervised Classification Model:** This model (`SelfSupervisedModel`) uses a shared encoder with multiple `TaskHead`s for solving simple classification problems based on the augmentations:
    *   **Task 1 (Augmented vs. Original):** A binary classification head that learns to distinguish between a normal snippet and one that has been temporally shuffled, reversed, or had its speed altered.
    *   **Task 2 (Rotation Prediction):** A multi-class classification head that predicts which rotation (0°, 90°, 180°, or 270°) was applied to a snippet.

### 3. Training Paradigms and Loss Functions

*   **Decoder Training:** The decoders are trained using a combined loss:
    *   **Intensity Loss (MSE):** A Mean Squared Error loss (`intensity_loss`) that measures the pixel-wise difference between the ground truth and the generated frames.
    *   **Gradient Loss:** This loss (`gradient_loss`) helps in preserving the sharpness and edge details in the reconstructed frames by comparing their image gradients.

*   **MoCoV2 Training:** The contrastive learning component is trained with an **InfoNCE (Noise Contrastive Estimation)** loss (`ContrastiveLoss`). This loss function encourages the model to learn representations that are invariant to the applied augmentations, bringing similar pairs closer and pushing dissimilar ones apart.

*   **Self-Supervised Classification:** The classification heads are trained using a standard **Cross-Entropy Loss** to optimize their performance on the pretext tasks.

### 4. Anomaly Detection and Scoring

During the testing phase, the trained models are used to calculate a "regularity score" for new video snippets. A low regularity score indicates a potential anomaly. This score is a composite of three metrics:

*   **S_Frame (Frame-based Regularity):** This score is derived from the Frame Prediction Decoder and is based on the **Peak Signal-to-Noise Ratio (PSNR)** between the predicted next frame and the actual ground truth frame. A high PSNR (low prediction error) corresponds to a high regularity score.

*   **S_Tcon (Contrastive Regularity):** Calculated using the trained `MoCoV2` model. It measures the cosine similarity between the feature representation of an original snippet and its various augmentations. A high similarity score indicates that the model understands the snippet's content, making it "regular."

*   **S_TClass (Classification-based Regularity):** This score is based on the confidence of the `SelfSupervisedModel`. A snippet is considered regular if the model can easily and correctly predict the transformations applied to it. The score is calculated as `1 - (average prediction error)`.

Finally, these three scores are combined and normalized to produce a final anomaly score for each snippet, providing a robust and multi-faceted measure of abnormality.

It is noteworthy that, when evaluated individually, the model based on **Contrastive Learning consistently outperformed the others**.
