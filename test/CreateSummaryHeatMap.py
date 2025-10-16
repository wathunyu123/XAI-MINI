import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- การตั้งค่า ---
# Path ไปยังโฟลเดอร์ที่เก็บไฟล์ผลลัพธ์ .json
RESULTS_DIR = "evaluation_results/results_json/"

def CreateHeatMap(results_directory, filename="heatmaps/evaluation_summary_heatmap.png"):
    """
    อ่านไฟล์ผลลัพธ์ JSON จากการประเมิน แล้วสร้าง Heatmap สรุปผลการทำงาน (.png)
    """
    print("\n--- Generating Final Summary Heatmap ---")
    
    # 1. ค้นหาไฟล์ผลลัพธ์ .json ทั้งหมด
    json_files = glob.glob(os.path.join(results_directory, "*_result.json"))
    if not json_files:
        print(f"Error: No result files found in '{results_directory}' to create a heatmap.")
        return

    # 2. กำหนด Keywords ทั้งหมดที่จะใช้เป็นคอลัมน์ใน Heatmap
    all_relevant_keywords = sorted([
        "anterior", "bone", "canine", "central incisor", "crown", "fracture", 
        "incisor", "lateral incisor", "lesion", "loss", "mandibular", "maxillary", 
        "normal", "pathology", "periapical", "restoration", "untreated"
    ])

    # 3. ประมวลผลไฟล์ JSON แต่ละไฟล์เพื่อสร้างข้อมูลสำหรับ Heatmap
    evaluation_data = []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f: result_data = json.load(f)
        case_name = result_data["case_name"]
        generated_narrative_lower = result_data.get("generated_narrative", "").lower()
        expected_keywords = set()
        try:
            with open(result_data["ground_truth_path"], 'r', encoding='utf-8') as gt_f:
                ground_truth = json.load(gt_f)
                expected_keywords = set(ground_truth.get("key_keywords_expected", []))
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found for {case_name}.")
        
        for keyword in all_relevant_keywords:
            score = 0  # 0 = Not Applicable (ไม่เกี่ยวข้อง)
            if keyword in generated_narrative_lower:
                score = 2  # 2 = Mentioned (พูดถึง)
            elif keyword in expected_keywords:
                score = 1  # 1 = Omission (มองข้าม)
            evaluation_data.append({"case": case_name, "keyword": keyword, "score": score})

    if not evaluation_data: print("No evaluation data to process for heatmap."); return
        
    # 4. แปลงข้อมูลให้อยู่ในรูปแบบ Matrix (DataFrame)
    df = pd.DataFrame(evaluation_data)
    heatmap_df = df.pivot(index="case", columns="keyword", values="score")
    
    # 5. สร้าง Heatmap Visualization
    cmap = ListedColormap(['#E0E0E0', '#FFC107', '#4CAF50']) # 0=เทา, 1=เหลือง, 2=เขียว
    score_labels = {0: "N/A", 1: "Omission", 2: "Mentioned"}
    annot_labels = heatmap_df.applymap(lambda x: score_labels.get(x, ""))
    plt.figure(figsize=(22, len(heatmap_df) * 0.8 + 2))
    ax = sns.heatmap(heatmap_df, annot=annot_labels, fmt="", cmap=cmap, linewidths=1.5,
                     linecolor='white', cbar=False, vmin=0, vmax=2)

    # 6. ปรับแต่งความสวยงามของกราฟ
    ax.set_title("Model Performance Summary", fontsize=20, pad=30)
    ax.set_xlabel("Clinical Keywords", fontsize=14); ax.set_ylabel("Test Cases", fontsize=14)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left'); plt.yticks(rotation=0)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Mentioned'),
                       Patch(facecolor='#FFC107', label='Omission (Missed)'),
                       Patch(facecolor='#E0E0E0', label='Not Applicable')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 7. บันทึกและแสดงผล
    save_path = os.path.join(os.path.dirname(results_directory), filename) # บันทึกไว้นอกโฟลเดอร์ results_json
    plt.savefig(save_path, dpi=300)
    print(f"\nSummary heatmap saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    CreateHeatMap(RESULTS_DIR)