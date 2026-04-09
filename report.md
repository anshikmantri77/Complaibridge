# Technical Architecture Diagram Analysis Report

## 1. Toolkits and Models Choice
The pipeline leverages a combination of deep learning and geometric computer vision:
- **EasyOCR (CRAFT + CRNN):** Chosen for text detection and recognition. CRAFT (Character Region Awareness for Text Detection) is particularly effective for diagrams where text might be isolated or uniquely positioned.
- **OpenCV:** Forms the backbone of the geometric analysis, managing Canny edge detection, Hough Line Transforms, and bitwise masking.
- **NetworkX:** Utilized for building and manipulating the topological graph once the vision pipeline extracts endpoints.
- **Matplotlib:** Used for generating the high-level relationship visualization.

### Choice Rationale
1.  **OCR Robustness:** Architecture diagrams often contain non-standard fonts and small labels sitting on dashed lines. EasyOCR outperforms standard Tesseract in these scenarios.
2.  **Deterministic Logic:** For structural diagrams, deterministic geometric association (containment and proximity) is preferred over end-to-end black-box models to ensure transparency and debuggability in entity nesting.

## 2. Advanced Pipeline Design
The production-grade pipeline implements several non-obvious optimizations:

### A. Entity-Masked Hough Transform
Standard line detection often fails because box borders (rectangles) generate stronger Hough peaks than internal arrows. The pipeline solves this by **pre-masking**: all detected text and boxes are zeroed out (set to black) before line detection begins, ensuring only true connections remain.

### B. Solid vs. Dashed Line Classification
We implement a **pixel fill-ratio sampling** algorithm. By sampling 20-50 localized points along a detected line's path in a binary mask, we calculate the ratio of 'on' pixels. 
- **Ratio > 0.65:** Classified as **Solid**.
- **Ratio < 0.65:** Classified as **Dashed**.
This is critical for distinguishing standard flows from specialized tunnels (like VPNs).

### C. Horizontal Token Merging
OCR often detects "Search UI" as two separate tokens. The pipeline includes a greedy horizontal merging pass that joins adjacent tokens with similar Y-coordinates and minimal X-gaps into single semantic entities.

### D. Multi-Pass Hierarchy Assignment
1.  **Pass 1:** Analyzes all quadrilaterals to determine group-in-group containment (e.g., Elasticsearch Serverless inside a Region).
2.  **Pass 2:** Assigns leaf nodes (text, icons) to their tightest-fitting parent container.
3.  **Label Propagation:** Groups that are unlabeled inherit the semantic name of text identified at their "header" (top boundary).

## 3. Heuristics and Trade-offs
- **Zone Classification:** A spatial heuristic is used to label top-level groups. Groups on the far left are tagged as "cloud", center-right as "external", and the middle zone as "onprem". This is a lightweight alternative to general scene understanding.
- **Endpoint Resolution:** Arrows are mapped to entities by calculating the shortest Euclidean distance from the line endpoint to an entity's axis-aligned bounding box (AABB).
- **Bidirectional Deduplication:** The pipeline recognizes reciprocal relationships (Flow A→B and Flow B→A) and consolidates them into a single bidirectional entity to simplify the resulting graph.

## 4. Future Considerations
- **Generic Icon Support:** While database cylinders are currently detected via aspect ratio, a small CNN (Convolutional Neural Network) could be integrated to classify a wider range of icons (Compute, Storage, Network).
- **Flow Directionality:** Currently, arrowheads are inferred by endpoint proximity. A template-matching approach for the 'head' triangle would increase accuracy for arrows that don't terminate exactly on a box border.
