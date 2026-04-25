from pathlib import Path
import json
import cv2
import time
from detect import load_detector
from ocr import build_ocr_engine
from pipeline import process_frame

def run_batch_test():
    print("[*] Booting KnightSight V2 AI Core...")
    detector = load_detector()
    ocr_engine = build_ocr_engine()
    
    test_dir = Path("eval_data/images")
    if not test_dir.exists():
        print("No eval_data/images folder found.")
        return
        
    image_paths = sorted(test_dir.glob("*.jpg"))
    if not image_paths:
        print("No images found.")
        return
        
    print(f"[*] Found {len(image_paths)} test images. Commencing Neural Extraction...")
    
    results = {}
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        annotated_img, results_json, _ = process_frame(img, detector, ocr_engine)
        
        # Save highest confidence plate
        if results_json:
            best_pred = max(results_json, key=lambda x: x.get("confidence", 0))
            results[img_path.name] = {
                "extracted_plate": best_pred.get("plate_text", "UNKNOWN"),
                "ocr_confidence": round(best_pred.get("confidence", 0), 4),
            }
            print(f"[{i+1}/{len(image_paths)}] {img_path.name} -> Plt: {best_pred.get('plate_text', 'UNKNOWN')} (Conf: {best_pred.get('confidence', 0):.2f})")
        else:
            results[img_path.name] = {"extracted_plate": "UNKNOWN", "ocr_confidence": 0}
            print(f"[{i+1}/{len(image_paths)}] {img_path.name} -> NO PLATE DETECTED")

    end_time = time.time()
    
    # Save the output for the judges
    out_file = Path("eval_data/Final_Test_Metrics.json")
    out_file.write_text(json.dumps(results, indent=4))
    
    print("\n" + "="*50)
    print(f"[*] BATCH EVALUATION COMPLETE")
    print(f"[*] Time Elapsed: {end_time - start_time:.2f} seconds")
    print(f"[*] Average Speed: {(end_time - start_time)/len(image_paths):.3f} seconds per image")
    print(f"[*] Final Test Data saved to: {out_file.absolute()}")
    print("="*50)

if __name__ == "__main__":
    run_batch_test()
