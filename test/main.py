import os
import json
import glob
from LLaVADentist import LLaVADentist
from CreateSummaryHeatMap import create_summary_heatmap
import cv2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- การตั้งค่า ---
BASE_MODEL_PATH = "llava-1.5-7b-hf-bnb-4bit"
ADAPTER_PATH = "adapter"

EVAL_DATA_DIR = "evaluation_dataset/anterior_teeth/"
OUTPUT_DIR = "evaluation_results/"
INSTRUCTION = "You are an expert specializing in dentistry. Describe the condition of the anterior teeth in this dentistry with clinical accuracy, mentioning any anatomy, pathology, or restorations."

# --- ฟังก์ชันหลักสำหรับรัน Evaluation ---
def main():
    # --- ขั้นตอนที่ 1: เตรียมโฟลเดอร์สำหรับเก็บผลลัพธ์ ---
    results_json_dir = os.path.join(OUTPUT_DIR, "results_json")
    heatmaps_dir = os.path.join(OUTPUT_DIR, "heatmaps")
    os.makedirs(results_json_dir, exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(heatmaps_dir, exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี

    # --- ขั้นตอนที่ 2: โหลดโมเดล ---
    # เรียกใช้ Class LLaVADentist โดยส่ง Path ของ Base Model และ Adapter เข้าไป
    # การโหลดโมเดลจะทำเพียงครั้งเดียว เพื่อประหยัดเวลา
    print("Initializing the model with PEFT...")
    model = LLaVADentist(BASE_MODEL_PATH, ADAPTER_PATH)
    print("-" * 30)

    # --- ขั้นตอนที่ 3: ค้นหาและประมวลผลข้อมูลทดสอบทีละเคส ---
    # ค้นหาไฟล์ .json ทั้งหมดที่อยู่ในโฟลเดอร์ EVAL_DATA_DIR
    ground_truth_files = glob.glob(os.path.join(EVAL_DATA_DIR, "*.json"))
    if not ground_truth_files:
        print(f"Error: No ground truth .json files found in '{EVAL_DATA_DIR}'.")
        return
    print(f"Found {len(ground_truth_files)} evaluation cases.")

    # วนลูปเพื่อทำงานกับไฟล์ .json แต่ละไฟล์
    for gt_path in ground_truth_files:
        # ดึงชื่อไฟล์ (ไม่รวมนามสกุล) เพื่อใช้เป็นชื่อเคส
        case_name = os.path.splitext(os.path.basename(gt_path))[0]
        print(f"\n--- Processing case: {case_name} ---")
        # เปิดและอ่านไฟล์ Ground Truth .json
        with open(gt_path, 'r', encoding='utf-8') as f: ground_truth = json.load(f)
        image_path = ground_truth.get("image_path")
        # ตรวจสอบว่าไฟล์รูปภาพที่ระบุใน .json มีอยู่จริงหรือไม่
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image path for '{case_name}' not found. Skipping.")
            continue
        
        # --- ขั้นตอนที่ 3.1: สร้างคำบรรยาย (Narrative) จากโมเดล ---
        print("Generating narrative...")
        generated_narrative = model.generate_narrative(image_path, INSTRUCTION)

        xai_explanations = {}
        # ดำเนินการสร้าง XAI ต่อเมื่อการสร้าง Narrative ไม่เกิดข้อผิดพลาด
        if "Error:" not in generated_narrative:
            print(f"Generated Narrative: {generated_narrative}")
            print("Generating XAI heatmaps...")
            # สร้าง Prompt ฉบับเต็ม (รวมคำตอบ) เพื่อใช้ในการคำนวณ Heatmap
            full_xai_prompt = f"USER: <image>\n{INSTRUCTION}\nASSISTANT: {generated_narrative}"
            
            # --- ขั้นตอนที่ 3.2: รวบรวม Keywords ที่จะใช้สร้าง Heatmap ---
            # ดึงคำทั้งหมดจาก Narrative ที่โมเดลสร้างขึ้น
            # ใช้ generated_narrative.lower() เพื่อให้การเปรียบเทียบไม่สนตัวพิมพ์เล็ก/ใหญ่
            keywords_in_narrative = generated_narrative.lower()
            # ดึงคำที่คาดหวัง (Expected) จากไฟล์ Ground Truth
            expected_keywords = ground_truth.get("key_keywords_expected", [])
            # สร้างลิสต์คำที่จะตรวจสอบทั้งหมด
            all_keywords_to_check = set(expected_keywords) | {word.strip(".,!?") for word in keywords_in_narrative.split()}
            
            # กำหนดลิสต์ของคำศัพท์ที่เกี่ยวข้องทางคลินิกที่เราสนใจ
            relevant_keywords = [
                "anterior", "bone", "canine", "central incisor", "crown", "fracture", 
                "incisor", "lateral incisor", "lesion", "loss", "mandibular", "maxillary", 
                "normal", "pathology", "periapical", "restoration", "untreated"
            ]
            # กรองเอาเฉพาะคำที่อยู่ในลิสต์ที่เราสนใจ เพื่อสร้าง Heatmap
            keywords_to_generate_heatmap_for = [kw for kw in relevant_keywords if kw in all_keywords_to_check]
            print(f"Keywords for heatmap generation: {keywords_to_generate_heatmap_for}")

            # วนลูปเพื่อสร้าง Heatmap สำหรับแต่ละ Keyword
            for keyword in keywords_to_generate_heatmap_for:
                was_mentioned = keyword in keywords_in_narrative
                print(f" - Generating for keyword: '{keyword}' (Mentioned: {was_mentioned})")
                
                # --- ขั้นตอนที่ 3.3: สร้างและบันทึก Heatmap ---
                heatmap = model.generate_xai_heatmap(image_path, full_xai_prompt, keyword)
                if heatmap is not None:
                    # ถ้าเป็นคำที่โมเดลไม่ได้พูดถึง (Omission) ให้เติม "_omitted" ต่อท้ายชื่อไฟล์
                    suffix = "" if was_mentioned else "_omitted"
                    heatmap_filename = f"{case_name}_heatmap_{keyword}{suffix}.jpg"
                    heatmap_path = os.path.join(heatmaps_dir, heatmap_filename)
                    # เรียกใช้ฟังก์ชัน superimpose_heatmap เพื่อสร้างภาพซ้อนทับและบันทึกเป็นไฟล์
                    cv2.imwrite(model.superimpose_heatmap(image_path, heatmap), heatmap_path)
                    # เก็บ Path ของไฟล์ Heatmap และสถานะการพูดถึงไว้ใน Dictionary
                    xai_explanations[keyword] = {"path": heatmap_path, "mentioned_in_narrative": was_mentioned}
                else: print(f"   - Failed to generate heatmap for '{keyword}'")
        else:
             print(f"Failed to generate narrative: {generated_narrative}")
             print("Skipping XAI heatmap generation.")

        # --- ขั้นตอนที่ 3.4: รวบรวมและบันทึกผลลัพธ์ทั้งหมดเป็นไฟล์ .json ---
        result_data = {
            "case_name": case_name, "image_path": image_path, "ground_truth_path": gt_path,
            "analysis_focus": ground_truth.get("analysis_focus", "N/A"),
            "generated_narrative": generated_narrative,
            "expert_narrative": ground_truth.get("expert_narrative", "N/A"),
            "xai_explanations": xai_explanations
        }
        result_filename = f"{case_name}_result.json"
        result_path = os.path.join(results_json_dir, result_filename)
        # บันทึก Dictionary ทั้งหมดลงในไฟล์ .json
        with open(result_path, 'w', encoding='utf-8') as f: json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"Saved results for {case_name} to {result_path}")
    
    print("\nEvaluation run completed.")
    
    # --- ขั้นตอนที่ 4: สร้าง Heatmap สรุปผลหลังจากประมวลผลทุกเคสเสร็จ ---
    # เรียกใช้ฟังก์ชันสร้าง heatmap ที่เราย้ายเข้ามาไว้ในไฟล์นี้
    create_summary_heatmap(results_json_dir)

if __name__ == "__main__":
    main()