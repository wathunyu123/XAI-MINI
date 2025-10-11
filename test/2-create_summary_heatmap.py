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

def create_summary_heatmap(results_directory, filename="evaluation_summary_heatmap.png"):
    """
    อ่านไฟล์ผลลัพธ์ JSON จากการประเมิน แล้วสร้าง Heatmap สรุปผลการทำงาน
    เวอร์ชันนี้ถูกปรับปรุงให้รองรับ Keyword ที่เป็นวลี (หลายคำ) ได้
    """
    
    # --- ขั้นตอนที่ 1: ค้นหาไฟล์ผลลัพธ์ทั้งหมด ---
    json_files = glob.glob(os.path.join(results_directory, "*_result.json"))
    if not json_files:
        print(f"Error: No result files found in '{results_directory}'.")
        print("Not Found Result!")
        return

    # --- ขั้นตอนที่ 2: กำหนด Keywords ทั้งหมดที่จะใช้เป็นคอลัมน์ใน Heatmap ---
    # การกำหนดลิสต์นี้ไว้ล่วงหน้า ทำให้ Heatmap ของทุกการรันมีคอลัมน์เหมือนกัน เปรียบเทียบได้ง่าย
    all_relevant_keywords = sorted([
        "anterior", "bone", "canine", "central incisor", "crown", "fracture", 
        "incisor", "lateral incisor", "lesion", "loss", "mandibular", "maxillary", 
        "normal", "pathology", "periapical", "restoration", "untreated"
    ])

    # --- ขั้นตอนที่ 3: ประมวลผลไฟล์ JSON แต่ละไฟล์เพื่อสร้างข้อมูลสำหรับ Heatmap ---
    evaluation_data = [] # สร้างลิสต์ว่างเพื่อเก็บข้อมูล
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        case_name = result_data["case_name"]
        # อ่าน Narrative ที่โมเดลสร้างขึ้น และแปลงเป็นตัวพิมพ์เล็กทั้งหมดเพื่อให้ง่ายต่อการเปรียบเทียบ
        generated_narrative_lower = result_data.get("generated_narrative", "").lower()

        # โหลดไฟล์ Ground Truth เพื่อหาว่าในเคสนี้ "ควรจะ" มี Keyword อะไรบ้าง
        expected_keywords = set()
        try:
            with open(result_data["ground_truth_path"], 'r', encoding='utf-8') as gt_f:
                ground_truth = json.load(gt_f)
                expected_keywords = set(ground_truth.get("key_keywords_expected", []))
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found for {case_name}. Omissions cannot be calculated.")
        
        # วนลูปตรวจสอบสถานะของ Keyword แต่ละคำสำหรับเคสนี้
        for keyword in all_relevant_keywords:
            score = 0  # 0 = Not Applicable (ไม่เกี่ยวข้อง) เป็นค่าเริ่มต้น
            
            # --- Logic การให้คะแนน ---
            # <<< การตรวจสอบให้รองรับวลี (หลายคำ) >>>
            # ตรวจสอบว่า keyword (ซึ่งอาจเป็นวลี) อยู่ใน narrative ที่สร้างขึ้นหรือไม่
            if keyword in generated_narrative_lower:
                score = 2  # 2 = Mentioned (พูดถึง)
            elif keyword in expected_keywords:
                # ถ้าโมเดลไม่ได้พูดถึง แต่เป็นคำที่ควรจะพูด (มีใน expected_keywords)
                score = 1  # 1 = Omission (มองข้าม)
            
            # เพิ่มข้อมูล (case, keyword, score) ลงในลิสต์
            evaluation_data.append({
                "case": case_name,
                "keyword": keyword,
                "score": score
            })

    # --- ขั้นตอนที่ 4: แปลงข้อมูลให้อยู่ในรูปแบบ Matrix (DataFrame) ---
    if not evaluation_data:
        print("No evaluation data could be processed.")
        return
        
    df = pd.DataFrame(evaluation_data)
    # ใช้ฟังก์ชัน pivot เพื่อเปลี่ยนข้อมูลจากรูปแบบยาว (long format) เป็นรูปแบบกว้าง (wide format)
    # โดยให้ 'case' เป็นแถว (index), 'keyword' เป็นคอลัมน์, และ 'score' เป็นค่าในแต่ละเซลล์
    heatmap_df = df.pivot(index="case", columns="keyword", values="score")
    
    # --- ขั้นตอนที่ 5: สร้าง Heatmap Visualization ---
    
    # กำหนดสีและข้อความสำหรับแต่ละค่าคะแนน
    cmap = ListedColormap(['#E0E0E0', '#FFC107', '#4CAF50']) # 0=เทา, 1=เหลือง, 2=เขียว
    score_labels = {0: "N/A", 1: "Omission", 2: "Mentioned"}
    # สร้าง Matrix ของข้อความที่จะแสดงในแต่ละเซลล์ (annot)
    annot_labels = heatmap_df.applymap(lambda x: score_labels.get(x, ""))

    # กำหนดขนาดของกราฟ (ความกว้าง 22, ความสูงปรับตามจำนวนเคส)
    plt.figure(figsize=(22, len(heatmap_df) * 0.8 + 2))
    
    # สร้าง Heatmap ด้วยไลบรารี Seaborn
    ax = sns.heatmap(
        heatmap_df,
        annot=annot_labels,  # ใช้ข้อความที่เรากำหนดเอง
        fmt="",              # ต้องเป็นค่าว่างเมื่อใช้ annot ที่เป็น DataFrame
        cmap=cmap,           # ใช้ชุดสีที่เรากำหนด
        linewidths=1.5,      # เพิ่มเส้นขอบสีขาวระหว่างเซลล์
        linecolor='white',
        cbar=False,          # ซ่อนแถบสี (Color bar) ด้านข้าง
        vmin=0,              # จุดเริ่มต้นของสเกลสี
        vmax=2               # บังคับให้สเกลสีเต็มช่วง 0-2 เสมอ เพื่อแก้ปัญหา Bug สีเพี้ยน
    )

    # --- การปรับแต่งความสวยงามของกราฟ ---
    ax.set_title("Model Performance Summary", fontsize=20, pad=30)
    ax.set_xlabel("Clinical Keywords", fontsize=14)
    ax.set_ylabel("Test Cases", fontsize=14)
    ax.xaxis.tick_top() # ย้ายชื่อคอลัมน์ (Keywords) ไปไว้ด้านบน
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left') # หมุนชื่อคอลัมน์ 45 องศาเพื่อให้อ่านง่าย
    plt.yticks(rotation=0)

    # สร้าง Legend (คำอธิบายสัญลักษณ์) ด้วยตัวเอง
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Mentioned'),
                       Patch(facecolor='#FFC107', label='Omission (Missed)'),
                       Patch(facecolor='#E0E0E0', label='Not Applicable')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    # ปรับ Layout เพื่อให้มีที่ว่างสำหรับ Legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # บันทึกกราฟเป็นไฟล์รูปภาพ
    plt.savefig(filename, dpi=300)
    print(f"\nSummary heatmap saved to '{filename}'")
    # แสดงกราฟขึ้นมาบนหน้าจอ
    plt.show()


if __name__ == "__main__":
    create_summary_heatmap(RESULTS_DIR)