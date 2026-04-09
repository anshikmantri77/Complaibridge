import cv2
import numpy as np
import easyocr
import json
import os
from typing import List, Dict, Any, Tuple

class DiagramAnalyzer:
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.input_path = os.path.join(workspace_root, "input", "diagram.png")
        self.output_dir = os.path.join(workspace_root, "output")
        
        self.image = cv2.imread(self.input_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {self.input_path}")
        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Use CPU for OCR to avoid CUDA issues in this environment
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.entities = []
        self.relationships = []
        self.id_counter = 0

    def generate_id(self):
        self.id_counter += 1
        return f"node_{self.id_counter}"

    def detect_entities(self):
        print("Starting Detection & Extraction...")
        
        # 1. OCR for text elements
        print("Running OCR...")
        ocr_results = self.reader.readtext(self.image)
        for (bbox, text, prob) in ocr_results:
            if prob < 0.3: continue
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
            
            self.entities.append({
                "id": self.generate_id(),
                "type": "text",
                "label": text,
                "bounding_box": {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min},
                "parent_id": None,
                "children": []
            })

        # 2. Box/Group Detection
        print("Detecting boxed regions...")
        # Use different thresholds to find nested boxes
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # Filter out too small or too large detections
                if 40 < w < self.image.shape[1] * 0.98 and 40 < h < self.image.shape[0] * 0.98:
                    # Check if it's already detected as a box (overlap)
                    is_new = True
                    for e in [ent for ent in self.entities if ent["type"] == "group"]:
                        eb = e["bounding_box"]
                        # If boxes are almost identical, skip
                        if abs(x - eb["x"]) < 10 and abs(y - eb["y"]) < 10 and abs(w - eb["w"]) < 10:
                            is_new = False
                            break
                    
                    if is_new:
                        self.entities.append({
                            "id": self.generate_id(),
                            "type": "group",
                            "label": "Region", 
                            "bounding_box": {"x": x, "y": y, "w": w, "h": h},
                            "parent_id": None,
                            "children": []
                        })

    def associate(self):
        print("Associating text/icons to groups...")
        groups = [e for e in self.entities if e["type"] == "group"]
        # Sort by area ascending so we find the "tightest" fit first
        groups.sort(key=lambda x: x["bounding_box"]["w"] * x["bounding_box"]["h"])
        
        non_groups = [e for e in self.entities if e["type"] != "group"]
        
        for item in non_groups:
            ix, iy = item["bounding_box"]["x"], item["bounding_box"]["y"]
            iw, ih = item["bounding_box"]["w"], item["bounding_box"]["h"]
            item_center = (ix + iw/2, iy + ih/2)
            
            for g in groups:
                gx, gy = g["bounding_box"]["x"], g["bounding_box"]["y"]
                gw, gh = g["bounding_box"]["w"], g["bounding_box"]["h"]
                
                if (gx <= item_center[0] <= gx + gw and 
                    gy <= item_center[1] <= gy + gh):
                    item["parent_id"] = g["id"]
                    g["children"].append(item["id"])
                    # If the group doesn't have a label yet and this text is likely its title
                    # Titles are usually at the top of the box
                    if item["type"] == "text" and (iy - gy) < 30:
                        g["label"] = item["label"]
                    break

        # Associate groups to groups (nested)
        for i, g_child in enumerate(groups):
            cx, cy = g_child["bounding_box"]["x"], g_child["bounding_box"]["y"]
            cw, ch = g_child["bounding_box"]["w"], g_child["bounding_box"]["h"]
            child_center = (cx + cw/2, cy + ch/2)
            
            for j, g_parent in enumerate(groups):
                if i == j: continue
                px, py = g_parent["bounding_box"]["x"], g_parent["bounding_box"]["y"]
                pw, ph = g_parent["bounding_box"]["w"], g_parent["bounding_box"]["h"]
                
                # Check containment
                if (px < cx and py < cy and 
                    px + pw > cx + cw and py + ph > cy + ch):
                    # We found a parent; since groups are sorted by area, 
                    # the first one we find moving up is the direct parent
                    g_child["parent_id"] = g_parent["id"]
                    g_parent["children"].append(g_child["id"])
                    break

    def detect_relationships(self):
        print("Detecting relationships (Arrows)...")
        # Isolate arrows: Threshold -> Filter out boxes/text regions
        mask = np.ones(self.gray.shape, dtype="uint8") * 255
        for e in self.entities:
            b = e["bounding_box"]
            cv2.rectangle(mask, (b["x"]-2, b["y"]-2), (b["x"]+b["w"]+4, b["y"]+b["h"]+4), 0, -1)
        
        only_lines = cv2.bitwise_and(self.gray, self.gray, mask=mask)
        _, binary = cv2.threshold(only_lines, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Hough Lines for arrows
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, 20, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Determine direction: Check which end is closer to a tip
                # For simplified assignment, we'll link nodes closest to endpoints
                src_id = self.find_closest_entity(x1, y1)
                dst_id = self.find_closest_entity(x2, y2)
                
                if src_id and dst_id and src_id != dst_id:
                    self.relationships.append({
                        "from": src_id,
                        "to": dst_id,
                        "direction": "unidirectional", # Default
                        "style": "solid", # Default
                        "label": None
                    })

    def find_closest_entity(self, x, y):
        # Find small entities (not parent containers) near point
        min_dist = 100
        best_id = None
        for e in self.entities:
            # Skip large container boxes, focus on leaf nodes or small boxes
            if e["type"] == "group" and e["bounding_box"]["w"] > self.image.shape[1] * 0.5:
                continue
            
            bx = e["bounding_box"]["x"]
            by = e["bounding_box"]["y"]
            bw = e["bounding_box"]["w"]
            bh = e["bounding_box"]["h"]
            
            # Distance from point to box boundary
            dx = max(bx - x, 0, x - (bx + bw))
            dy = max(by - y, 0, y - (by + bh))
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                best_id = e["id"]
        return best_id

    def run(self):
        self.detect_entities()
        self.associate()
        self.detect_relationships()
        
        # Save Outputs
        os.makedirs(self.output_dir, exist_ok=True)
        
        # JSON
        with open(os.path.join(self.output_dir, "structure.json"), "w") as f:
            json.dump({"entities": self.entities, "relationships": self.relationships}, f, indent=2)
            
        # Annotated Image
        annotated = self.image.copy()
        colors = {"text": (255, 0, 0), "group": (0, 255, 0), "icon": (0, 165, 255), "arrow": (0, 0, 255)}
        for e in self.entities:
            b = e["bounding_box"]
            cv2.rectangle(annotated, (b["x"], b["y"]), (b["x"]+b["w"], b["y"]+b["h"]), colors.get(e["type"], (255,255,255)), 2)
            cv2.putText(annotated, f"{e['type']}: {e['label'][:15]}", (b["x"], b["y"]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
        # Draw relationships (simple lines for demo)
        for r in self.relationships:
            src = next(e for e in self.entities if e["id"] == r["from"])
            dst = next(e for e in self.entities if e["id"] == r["to"])
            s_b = src["bounding_box"]
            d_b = dst["bounding_box"]
            cv2.line(annotated, (s_b["x"]+s_b["w"]//2, s_b["y"]+s_b["h"]//2), 
                                (d_b["x"]+d_b["w"]//2, d_b["y"]+d_b["h"]//2), (0,0,255), 2)
            
        cv2.imwrite(os.path.join(self.output_dir, "annotated_diagram.png"), annotated)
        print(f"Results saved to {self.output_dir}")

if __name__ == "__main__":
    analyzer = DiagramAnalyzer(r"c:\Users\Administrator\.gemini\antigravity\scratch\complianceai")
    analyzer.run()
