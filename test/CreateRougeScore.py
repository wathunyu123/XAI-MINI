import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- การตั้งค่า ---
# Path ไปยังโฟลเดอร์ที่สคริปต์หลักใช้บันทึกผลลัพธ์ .json
RESULTS_DIR = "evaluation_results/results_json/"

class CreateRougeMatrix:
    """
    คลาสสำหรับอ่านค่า ROUGE score ที่คำนวณไว้ล่วงหน้าจากไฟล์ผลลัพธ์
    และสร้าง Heatmap สรุปผลเพื่อแสดงเป็นภาพ
    """
    def __init__(self, results_directory):
        """
        เมธอด Constructor ของคลาส จะถูกเรียกใช้เมื่อมีการสร้าง Object
        ทำหน้าที่รับ Path ไปยังโฟลเดอร์ผลลัพธ์
        """
        self.results_directory = results_directory
        print("Initialized ROUGE Matrix Creator.")

    def _load_scores_from_json(self):
        """
        Private method ทำหน้าที่ค้นหาไฟล์ผลลัพธ์ .json ทั้งหมด และดึงค่า ROUGE score ออกมา
        """
        # ค้นหาไฟล์ .json ทั้งหมดที่อยู่ในโฟลเดอร์ที่กำหนด
        json_files = glob.glob(os.path.join(self.results_directory, "*_result.json"))
        if not json_files:
            print(f"Error: No result files found in '{self.results_directory}'.")
            print("Please run your main.py script first to generate results.")
            return None

        all_scores = [] # สร้างลิสต์ว่างเพื่อเก็บคะแนนจากทุกเคส
        print(f"Found {len(json_files)} result files. Extracting ROUGE scores...")

        # วนลูปเพื่ออ่านไฟล์ .json แต่ละไฟล์
        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            case_name = data.get("case_name", "Unknown Case")
            # ดึงค่า ROUGE score จาก Dictionary ของข้อมูลที่อ่านมา
            rouge_scores = data.get("rouge_scores")

            # ประมวลผลเฉพาะเคสที่มีข้อมูล ROUGE score ที่ถูกต้องเท่านั้น
            if rouge_scores and isinstance(rouge_scores, dict):
                # เพิ่มชื่อเคสเข้าไปใน Dictionary ของคะแนน เพื่อใช้เป็น index ในภายหลัง
                rouge_scores['case'] = case_name
                all_scores.append(rouge_scores)
            else:
                print(f"Skipping case '{case_name}' as it has no valid ROUGE scores in its result file.")
        
        return all_scores

    def _create_heatmap_dataframe(self, scores_data):
        """
        Private method ทำหน้าที่แปลงลิสต์ของ Dictionary คะแนน ให้กลายเป็น
        Pandas DataFrame ซึ่งเป็นรูปแบบที่เหมาะสำหรับสร้าง Heatmap
        """
        if not scores_data:
            return None
        
        # สร้าง DataFrame จากลิสต์ของ Dictionary
        df = pd.DataFrame(scores_data)
        
        # กำหนดให้คอลัมน์ 'case' เป็น index (ชื่อแถวของ Heatmap)
        df.set_index('case', inplace=True)
        
        # เลือกมาเฉพาะคอลัมน์ ROUGE score ที่สำคัญที่สุดเพื่อนำไปสร้าง Heatmap
        # (ค่าเหล่านี้โดยทั่วไปคือ F1-score ที่คำนวณโดยไลบรารี 'evaluate')
        heatmap_df = df[['rouge1', 'rouge2', 'rougeL']]
        
        return heatmap_df

    def generate(self, filename="rouge_scores_heatmap.png"):
        """
        Public method หลัก ทำหน้าที่ควบคุมกระบวนการทั้งหมด ตั้งแต่การโหลดข้อมูล,
        สร้าง DataFrame, และพล็อต Heatmap
        
        Args:
            filename (str): ชื่อไฟล์รูปภาพผลลัพธ์ที่ต้องการบันทึก
        """
        # เรียกใช้เมธอดภายในเพื่อโหลดคะแนน
        scores_data = self._load_scores_from_json()
        
        # ตรวจสอบว่ามีข้อมูลพร้อมสำหรับสร้าง Heatmap หรือไม่
        if scores_data is None or not scores_data:
            print("No valid data available to generate a ROUGE heatmap.")
            return

        # เรียกใช้เมธอดภายในเพื่อสร้าง DataFrame
        heatmap_df = self._create_heatmap_dataframe(scores_data)
        
        if heatmap_df is None or heatmap_df.empty:
            print("Could not create DataFrame for the heatmap.")
            return

        print("\nGenerating ROUGE score heatmap...")
        
        # --- ส่วนของการพล็อต Heatmap ---
        # กำหนดขนาดของกราฟ โดยปรับความสูงตามจำนวนเคส
        plt.figure(figsize=(10, len(heatmap_df) * 0.6 + 2))
        
        # สร้าง Heatmap ด้วยไลบรารี Seaborn
        ax = sns.heatmap(
            heatmap_df,
            annot=True,          # แสดงค่าคะแนนในแต่ละเซลล์
            fmt=".3f",           # จัดรูปแบบทศนิยม 3 ตำแหน่งเพื่อความแม่นยำ
            cmap="YlGnBu",       # ชุดสีที่สวยงามและเป็นมิตรกับผู้ที่ตาบอดสี (เหลือง-เขียว-น้ำเงิน)
            linewidths=.5,       # เพิ่มเส้นขอบสีขาวระหว่างเซลล์
            linecolor='white',
            vmin=0.0,            # ล็อกสเกลสีให้เริ่มต้นที่ 0.0 (แย่ที่สุด)
            vmax=1.0,            # และสิ้นสุดที่ 1.0 (สมบูรณ์แบบ) เพื่อให้สีมีความสม่ำเสมอทุกครั้งที่รัน
            cbar_kws={'label': 'ROUGE F1-Score (0 to 1)'} # เพิ่มป้ายกำกับให้แถบสี
        )

        # --- การปรับแต่งความสวยงามของกราฟ ---
        ax.set_title("ROUGE Score Performance Matrix", fontsize=16, pad=20)
        ax.set_xlabel("ROUGE Metric Type", fontsize=12)
        ax.set_ylabel("Test Cases", fontsize=12)
        ax.xaxis.tick_top() # ย้ายป้ายกำกับแกน X ไปไว้ด้านบน
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # ปรับ Layout ให้อัตโนมัติ
        plt.tight_layout()
        
        # บันทึกกราฟเป็นไฟล์รูปภาพในโฟลเดอร์ evaluation_results/
        save_path = os.path.join(os.path.dirname(self.results_directory), filename)
        plt.savefig(save_path, dpi=300)
        print(f"ROUGE score heatmap saved to '{save_path}'")
        
        # แสดงกราฟขึ้นมาบนหน้าจอ
        plt.show()

if __name__ == "__main__":
    # บล็อกนี้ทำให้สคริปต์สามารถรันได้โดยตรงจาก Command Line
    
    # ตรวจสอบว่ามี Library ที่จำเป็นสำหรับการแสดงผลติดตั้งอยู่หรือไม่
    try:
        import pandas
        import seaborn
        import matplotlib
    except ImportError:
        print("\n--- Installation required ---")
        print("Please run: pip install pandas seaborn matplotlib")
        exit()

    # สร้าง instance ของคลาส และเรียกใช้กระบวนการสร้าง Heatmap
    matrix_creator = CreateRougeMatrix(RESULTS_DIR)
    matrix_creator.generate()