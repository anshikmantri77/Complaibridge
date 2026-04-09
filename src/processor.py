import cv2
import numpy as np
import easyocr
import json
import os
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt

# B1. TYPED DATACLASSES
@dataclass
class BBox:
    x: int; y: int; w: int; h: int

    @property
    def cx(self): return self.x + self.w // 2

    @property
    def cy(self): return self.y + self.h // 2

    @property
    def area(self): return self.w * self.h

    def contains(self, other: "BBox") -> bool:
        return (self.x <= other.x and
                self.y <= other.y and
                self.x + self.w >= other.x + other.w and
                self.y + self.h >= other.y + other.h)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

@dataclass
class Entity:
    id: str
    type: str           # "group" | "text" | "icon"
    label: str
    bbox: BBox
    confidence: float = 1.0
    parent_id: Optional[str] = None
    children: list = field(default_factory=list)
    zone_hint: Optional[str] = None

@dataclass
class Relationship:
    from_id: str
    to_id: str
    direction: str      # "unidirectional" | "bidirectional"
    style: str          # "solid" | "dashed"
    label: Optional[str] = None

class DiagramAnalyzer:
    def __init__(self, img_path: str, output_dir: str):
        self.img_path = img_path
        self.output_dir = output_dir
        self.image = cv2.imread(img_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {img_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.entities: List[Entity] = []
        self.relationships: List[Relationship] = []
        self.id_counter = 0
        self.id_map: Dict[str, Entity] = {}

    def generate_id(self, prefix="ent"):
        self.id_counter += 1
        return f"{prefix}_{self.id_counter}"

    def preprocess(self):
        pass

    def detect_boxes(self):
        edges = cv2.Canny(self.gray, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_area = self.image.shape[0] * self.image.shape[1]
        seen = set()

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(cnt)

            if len(approx) < 4: continue
            if w * h < 1500: continue
            if w * h > 0.85 * img_area: continue
            if w < 30 or h < 20: continue
            
            key = (x//10, y//10, w//10, h//10)
            if key in seen: continue
            seen.add(key)

            rect_area = w * h
            contour_area = cv2.contourArea(cnt)
            confidence = min(1.0, contour_area / rect_area if rect_area > 0 else 0)

            bbox = BBox(x, y, w, h)
            entity = Entity(
                id=self.generate_id("group"),
                type="group",
                label="Region",
                bbox=bbox,
                confidence=float(confidence)
            )
            self.entities.append(entity)
            self.id_map[entity.id] = entity

    def extract_text(self):
        results = self.reader.readtext(self.image)
        for (bbox_pts, text, confidence) in results:
            if confidence < 0.3: continue
            x_min = int(min(p[0] for p in bbox_pts))
            y_min = int(min(p[1] for p in bbox_pts))
            x_max = int(max(p[0] for p in bbox_pts))
            y_max = int(max(p[1] for p in bbox_pts))
            
            bbox = BBox(x_min, y_min, x_max - x_min, y_max - y_min)
            entity = Entity(
                id=self.generate_id("text"),
                type="text",
                label=text,
                bbox=bbox,
                confidence=float(confidence)
            )
            self.entities.append(entity)
            self.id_map[entity.id] = entity

    def merge_text_tokens(self, x_gap=60, y_gap=12):
        text_entities = [e for e in self.entities if e.type == "text"]
        if not text_entities: return

        processed = set()
        text_entities.sort(key=lambda e: (e.bbox.y, e.bbox.x))
        new_entities = []

        for i, e1 in enumerate(text_entities):
            if e1.id in processed: continue
            current_phrase = [e1]
            processed.add(e1.id)
            
            changed = True
            while changed:
                changed = False
                for j, e2 in enumerate(text_entities):
                    if e2.id in processed: continue
                    last_e = current_phrase[-1]
                    avg_h = (last_e.bbox.h + e2.bbox.h) / 2
                    if (abs(last_e.bbox.cy - e2.bbox.cy) < avg_h + y_gap and
                        (e2.bbox.x - (last_e.bbox.x + last_e.bbox.w)) < x_gap and
                        (e2.bbox.x - (last_e.bbox.x + last_e.bbox.w)) > -10):
                        current_phrase.append(e2)
                        processed.add(e2.id)
                        changed = True
                        break
            
            if len(current_phrase) > 1:
                x_min = min(e.bbox.x for e in current_phrase)
                y_min = min(e.bbox.y for e in current_phrase)
                x_max = max(e.bbox.x + e.bbox.w for e in current_phrase)
                y_max = max(e.bbox.y + e.bbox.h for e in current_phrase)
                merged_label = " ".join(e.label for e in current_phrase)
                avg_conf = sum(e.confidence for e in current_phrase) / len(current_phrase)
                
                merged_entity = Entity(
                    id=self.generate_id("text_merged"),
                    type="text",
                    label=merged_label,
                    bbox=BBox(x_min, y_min, x_max - x_min, y_max - y_min),
                    confidence=avg_conf
                )
                new_entities.append(merged_entity)
            else:
                new_entities.append(e1)

        self.entities = [e for e in self.entities if e.type != "text"] + new_entities
        self.id_map = {e.id: e for e in self.entities}

    def detect_icons(self):
        # Database cylinder detection
        contours, _ = cv2.findContours(self.gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h > 0 else 0
            if 0.3 <= ar <= 1.0 and 800 < w*h < 10000:
                overlap = False
                for e in self.entities:
                    if e.bbox.contains(BBox(x, y, w, h)): 
                        if e.type == "group": continue
                        overlap = True
                        break
                if not overlap:
                    icon = Entity(
                        id=self.generate_id("icon"),
                        type="icon",
                        label="Database",
                        bbox=BBox(x, y, w, h),
                        confidence=0.75
                    )
                    self.entities.append(icon)
                    self.id_map[icon.id] = icon

    def detect_arrows(self):
        mask = np.ones(self.gray.shape, dtype="uint8") * 255
        for e in self.entities:
            b = e.bbox
            cv2.rectangle(mask, (b.x - 2, b.y - 2), (b.x + b.w + 4, b.y + b.h + 4), 0, -1)
        
        clean_for_hough = cv2.bitwise_and(self.gray, self.gray, mask=mask)
        edges = cv2.Canny(clean_for_hough, 30, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=15)
        
        if lines is not None:
            _, binary = cv2.threshold(clean_for_hough, 200, 255, cv2.THRESH_BINARY_INV)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                style = self.classify_line_style(x1, y1, x2, y2, binary)
                src_id = self.find_closest_entity(x1, y1)
                dst_id = self.find_closest_entity(x2, y2)
                
                if src_id and dst_id and src_id != dst_id:
                    rel = Relationship(from_id=src_id, to_id=dst_id, direction="unidirectional", style=style)
                    self.relationships.append(rel)

    def classify_line_style(self, x1, y1, x2, y2, binary_img) -> str:
        length = math.hypot(x2 - x1, y2 - y1)
        n_samples = max(int(length / 5), 1)
        filled = 0
        h, w = binary_img.shape
        for k in range(n_samples):
            t = k / max(n_samples - 1, 1)
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if 0 <= py < h and 0 <= px < w:
                if binary_img[py, px] > 0: filled += 1
        fill_ratio = filled / n_samples
        return "solid" if fill_ratio > 0.65 else "dashed"

    def find_closest_entity(self, x, y):
        min_dist = 80
        best_id = None
        for e in self.entities:
            if e.bbox.w > self.image.shape[1] * 0.5: continue
            dx = max(e.bbox.x - x, 0, x - (e.bbox.x + e.bbox.w))
            dy = max(e.bbox.y - y, 0, y - (e.bbox.y + e.bbox.h))
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                best_id = e.id
        return best_id

    def assign_hierarchy(self):
        groups = [e for e in self.entities if e.type == "group"]
        groups.sort(key=lambda g: g.bbox.area)

        for child in groups:
            for parent in groups:
                if child.id == parent.id: continue
                if parent.bbox.contains(child.bbox):
                    child.parent_id = parent.id
                    parent.children.append(child.id)
                    break

        leaves = [e for e in self.entities if e.type != "group"]
        for leaf in leaves:
            for group in groups:
                if group.bbox.contains(leaf.bbox):
                    leaf.parent_id = group.id
                    group.children.append(leaf.id)
                    if leaf.type == "text" and (leaf.bbox.y - group.bbox.y) < 30:
                        if group.label == "Region": group.label = leaf.label
                    break

        img_w = self.image.shape[1]
        for e in self.entities:
            if e.type == "group" and e.parent_id is None:
                if e.bbox.cx < img_w * 0.4: e.zone_hint = "cloud"
                elif e.bbox.cx < img_w * 0.7: e.zone_hint = "onprem"
                else: e.zone_hint = "external"

    def extract_arrow_labels(self):
        text_entities = [e for e in self.entities if e.type == "text"]
        for r in self.relationships:
            src, dst = self.id_map.get(r.from_id), self.id_map.get(r.to_id)
            if not src or not dst: continue
            mid_x, mid_y = (src.bbox.cx + dst.bbox.cx) // 2, (src.bbox.cy + dst.bbox.cy) // 2
            for t in text_entities:
                if abs(t.bbox.cx - mid_x) < 40 and abs(t.bbox.cy - mid_y) < 20:
                    r.label = t.label
                    break

    def deduplicate_relationships(self):
        seen, merged = {}, []
        for r in self.relationships:
            key_fwd, key_rev = (r.from_id, r.to_id), (r.to_id, r.from_id)
            if key_rev in seen: seen[key_rev].direction = "bidirectional"
            else:
                seen[key_fwd] = r
                merged.append(r)
        self.relationships = merged

    def export_json(self):
        output = {
            "entities": [],
            "relationships": [],
            "metadata": {
                "image_width": self.image.shape[1],
                "image_height": self.image.shape[0],
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships)
            }
        }
        for e in self.entities:
            output["entities"].append({
                "id": e.id, "type": e.type, "label": e.label, "confidence": round(e.confidence, 2),
                "bounding_box": e.bbox.to_dict(), "parent_id": e.parent_id, "children": e.children, "zone_hint": e.zone_hint
            })
        for r in self.relationships:
            output["relationships"].append({
                "from": r.from_id, "to": r.to_id, "direction": r.direction, "style": r.style, "label": r.label
            })
        with open(os.path.join(self.output_dir, "structure.json"), "w") as f:
            json.dump(output, f, indent=2)

    def draw_annotated_image(self):
        annotated = self.image.copy()
        colors = {"group": (94, 197, 34), "text": (246, 130, 59), "icon": (22, 115, 249)}
        for e in sorted(self.entities, key=lambda x: x.bbox.area, reverse=True):
            b = e.bbox
            color = colors.get(e.type, (255, 255, 255))
            cv2.rectangle(annotated, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
            cv2.putText(annotated, e.label[:15], (b.x, b.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for r in self.relationships:
            src, dst = self.id_map.get(r.from_id), self.id_map.get(r.to_id)
            if src and dst:
                cv2.arrowedLine(annotated, (src.bbox.cx, src.bbox.cy), (dst.bbox.cx, dst.bbox.cy), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(self.output_dir, "annotated_diagram.png"), annotated)

    def build_graph(self):
        self.G = nx.DiGraph()
        for e in self.entities: self.G.add_node(e.id, label=e.label, zone=e.zone_hint)
        for r in self.relationships:
            self.G.add_edge(r.from_id, r.to_id, style=r.style, label=r.label or "")

    def draw_graph_image(self):
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, labels={n: self.G.nodes[n]['label'][:10] for n in self.G.nodes}, 
                node_size=1000, node_color='lightblue', font_size=8)
        plt.savefig(os.path.join(self.output_dir, "relationship_graph.png"))
        plt.close()

    def run(self):
        print("Starting Pipeline...")
        self.detect_boxes()
        self.extract_text()
        self.merge_text_tokens()
        self.detect_icons()
        self.detect_arrows()
        self.assign_hierarchy()
        self.extract_arrow_labels()
        self.deduplicate_relationships()
        self.export_json()
        self.draw_annotated_image()
        self.build_graph()
        self.draw_graph_image()
        print(f"Done. Outputs in {self.output_dir}")

if __name__ == "__main__":
    import sys
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, "input", "diagram.png")
    output_path = os.path.join(base_path, "output")
    
    if not os.path.exists(input_path):
        print(f"Error: Input not found at {input_path}")
        sys.exit(1)
        
    analyzer = DiagramAnalyzer(input_path, output_path)
    analyzer.run()
