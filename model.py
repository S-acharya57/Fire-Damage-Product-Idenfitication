import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw

# Load YOLO model
class YOLOModel:
    def __init__(self, model_path="yolov5s.pt"):
        """
        Initialize the YOLO model. Downloads YOLOv5 pretrained model if not available.
        """
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
        print(f'YOLO Model:\n\n{self.model}')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        print(f'CLIP Model:\n\n{self.clip_model}')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.category_brands = {
            "electronics": ["Samsung", "Apple", "Sony", "LG", "Panasonic"],
            "furniture": ["Ikea", "Ashley", "La-Z-Boy", "Wayfair", "West Elm"],
            "appliances": ["Whirlpool", "GE", "Samsung", "LG", "Bosch"],
            "vehicles": ["Tesla", "Toyota", "Ford", "Honda", "Chevrolet"],
            "chair": ["Ikea", "Ashley", "Wayfair", "La-Z-Boy", "Herman Miller"],
            "microwave": ["Samsung", "Panasonic", "Sharp", "LG", "Whirlpool"],
            "table": ["Ikea", "Wayfair", "Ashley", "CB2", "West Elm"],
            "oven": ["Whirlpool", "GE", "Samsung", "Bosch", "LG"],
            "potted plant": ["The Sill", "PlantVine", "Lowe's", "Home Depot", "UrbanStems"],
            "couch": ["Ikea", "Ashley", "Wayfair", "La-Z-Boy", "CushionCo"],
            "cow": ["Angus", "Hereford", "Jersey", "Holstein", "Charolais"],
            "bed": ["Tempur-Pedic", "Ikea", "Sealy", "Serta", "Sleep Number"],
            "tv": ["Samsung", "LG", "Sony", "Vizio", "TCL"],
            "bin": ["Rubbermaid", "Sterilite", "Hefty", "Glad", "Simplehuman"],
            "refrigerator": ["Whirlpool", "GE", "Samsung", "LG", "Bosch"],
            "laptop": ["Dell", "HP", "Apple", "Lenovo", "Asus"],
            "smartphone": ["Apple", "Samsung", "Google", "OnePlus", "Huawei"],
            "camera": ["Canon", "Nikon", "Sony", "Fujifilm", "Panasonic"],
            "toaster": ["Breville", "Cuisinart", "Black+Decker", "Hamilton Beach", "Oster"],
            "fan": ["Dyson", "Honeywell", "Lasko", "Vornado", "Bionaire"],
            "vacuum cleaner": ["Dyson", "Shark", "Roomba", "Hoover", "Bissell"]
        }



    def predict_clip(self, image, brand_names):
        """
        Predict the most probable brand using CLIP.
        """
        inputs = self.clip_processor(
            text=brand_names,
            images=image,
            return_tensors="pt",
            padding=True
        )
        print(f'Inputs to clip processor:{inputs}')
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
        best_idx = probs.argmax().item()
        return brand_names[best_idx], probs[0, best_idx].item()


    def predict(self, image_path):
        """
        Run YOLO inference on an image.

        :param image_path: Path to the input image
        :return: List of predictions with labels and bounding boxes
        """
        results = self.model(image_path)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        predictions = results.pandas().xyxy[0]  # Get predictions as pandas DataFrame
        print(f'YOLO predictions:\n\n{predictions}')
        output = []
        for _, row in predictions.iterrows():
            category = row['name']
            confidence = row['confidence']
            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

            # Crop the detected region
            cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            # Match category to possible brands
            if category in self.category_brands:
                possible_brands = self.category_brands[category]
                print(f'Predicting with CLIP:\n\n')
                predicted_brand, clip_confidence = self.predict_clip(cropped_image, possible_brands)
            else:
                predicted_brand, clip_confidence = "Unknown", 0.0


            print(f'Predicted brand: {predicted_brand}')
            # Draw bounding box and label on the image
            draw.rectangle(bbox, outline="red", width=3)
            draw.text(
                (bbox[0], bbox[1] - 10),
                f"{predicted_brand} ({clip_confidence:.2f})",
                fill="red"
            )

            # Append result
            output.append({
                "category": category,
                "bbox": bbox,
                "confidence": confidence,
                "predicted_brand": predicted_brand,
                "clip_confidence": clip_confidence
            })
        return output
