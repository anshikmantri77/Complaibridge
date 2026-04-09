# Computer Vision Diagram Understanding Report

## 1. Toolkits and Models Choice
The solution utilizes the following core technologies:
- **OpenCV (Open Source Computer Vision Library):** Used for image preprocessing, geometric shape detection (contours, rectangles), and morphological operations to isolate lines and arrows.
- **EasyOCR:** Selected for text extraction due to its robustness in recognizing text within structured diagrams and its ability to return precise bounding boxes. It uses a CRAFT text detector and a CRNN recognition model.
- **NumPy:** Used for efficient array manipulations during image masking and geometric calculations.
- **Python's standard JSON library:** For exporting the structured logical graph.

### Rationale
- **OpenCV** provides low-level control over image feature extraction, which is essential for distinguishing between dashed lines, boxes, and icons.
- **EasyOCR** is preferred over Tesseract as it handles varied font styles and orientations often found in architecture diagrams more gracefully without extensive configuration.

## 2. Pipeline Design
The pipeline follows a sequential multi-stage approach:

1.  **Preprocessing:**
    - The input image is converted to grayscale to reduce dimensionality.
    - Gaussian blurring is applied to reduce noise for edge detection.
2.  **Detection & Extraction:**
    - **OCR Stage:** EasyOCR sweeps the image to identify all text elements and their coordinates.
    - **Contour Analysis:** Canny edge detection followed by contour approximation (`approxPolyDP`) is used to identify quadrilaterals (grouping boxes).
    - **Feature Masking:** Once text and boxes are identified, they are masked out to isolate the connecting lines and arrows.
3.  **Association Strategy:**
    - A **Geometric Containment** algorithm is used to build hierarchies. Entities (text, icons) are assigned to the smallest containing box.
    - Boxes are nested using a recursive containment check to reconstruct the architecture's "parent-child" relationships (e.g., Front-end inside Plant An App).
4.  **Relationship Mapping:**
    - **Hough Line Transform** detects straight edges representing connections.
    - The endpoints of these lines are associated with the nearest logical entity to form the directed graph.
5.  **Output Generation:**
    - The results are serialized into the requested JSON schema and visualized on an annotated image.

## 3. Trade-offs and Alternative Approaches
- **Rule-based vs. ML-based Arrow Detection:** While ML models (like YOLO) could be trained to find arrows, a rule-based approach using line endpoints was chosen for this assignment for its speed and lack of requirement for a specialized dataset. However, it may struggle with very complex overlapping lines.
- **Nested Box Complexity:** The current containment logic assumes boxes are perfectly rectangular and non-overlapping unless they are nested. For diagrams with free-form groupings, a pixel-wise semantic segmentation model (like Detectron2) would be a more robust but computationally expensive alternative.
- **OCR Accuracy:** Small labels (like "-VPN") can sometimes be missed if the threshold is too high. The pipeline uses a lower probability threshold for OCR to ensure these critical labels are captured, even at the cost of some noise.
