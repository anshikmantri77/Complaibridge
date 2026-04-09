# Production Architecture Diagram Analysis Pipeline

A high-fidelity computer vision pipeline designed to analyze architecture diagrams (PNG/JPG) and extract their complete logical structure — including entities, nested hierarchies, icons, and directional relationships.

## 🚀 Key Features
- **Intelligent Text Processing:** Uses **EasyOCR** with phrase-level token merging for coherent labels.
- **Robust Entity Detection:** Handles overlapping boxes, rounded corners, and structural groups.
- **Advanced Relationship Logic:**
  - **Entity Masking:** Prevents false arrow detections from box borders.
  - **Line Classification:** Distinguishes between **Solid** and **Dashed** lines (e.g., VPN tunnels).
  - **Bidirectional Detection:** Automatically merges reciprocal flows into single bidirectional arrows.
- **Structural Intelligence:**
  - **Multi-Pass Hierarchy:** Deeply nested containment analysis (e.g., *Service* inside *Cluster* inside *Cloud*).
  - **Zone Classification:** Heuristic-based labeling (Cloud, On-prem, External).
  - **Arrow Labeling:** Contextual proximity matching for floating labels (like "-VPN").

## 🛠 Prerequisites
- Python 3.8+
- OpenCV
- EasyOCR
- NetworkX & Matplotlib (for visualization)
- NumPy

## 📥 Installation
Install dependencies via pip:
```bash
pip install -r requirements.txt
```

## 🏃 How to Run
1.  **Input:** Place your architecture diagram in `input/diagram.png`.
2.  **Execute:**
    ```bash
    python src/processor.py
    ```
3.  **Outputs:**
    - `output/structure.json`: The fully structured logical graph.
    - `output/annotated_diagram.png`: Visualization of detected bounding boxes and relationships.
    - `output/relationship_graph.png`: NetworkX-generated topological visualization.

## 📁 Project Structure
- `src/processor.py`: Unified production pipeline.
- `input/`: Standard input directory.
- `output/`: Processed results and visualizations.
- `requirements.txt`: Pinned dependency versions.
- `report.md`: Detailed technical analysis.
