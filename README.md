# Diagram Understanding CV Pipeline

This project analyzes architecture diagrams and extracts their logical structure into JSON and annotated images.

## Prerequisites
- Python 3.8+
- OpenCV-headless (or standard OpenCV)
- EasyOCR
- NumPy

## Installation
Install the required dependencies using pip:

```bash
pip install opencv-python easyocr numpy matplotlib
```

## How to Run

1.  **Input:** Place your diagram image in the `input/` directory and name it `diagram.png`.
2.  **Execute:** Run the analyzer script:
    ```bash
    python src/processor.py
    ```
3.  **Output:**
    - `output/structure.json`: The logically reconstructed graph.
    - `output/annotated_diagram.png`: The original image with detection overlays.

## Project Structure
- `src/processor.py`: The core computer vision logic.
- `input/`: Directory for input images.
- `output/`: Directory where results are saved.
- `report.md`: Technical report on the pipeline design.
