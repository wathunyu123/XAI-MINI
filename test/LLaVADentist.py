import os
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# สร้างตัวแปร Global เพื่อเก็บข้อมูลที่ดักจับได้จาก Layer ของโมเดล
feature_maps = [] # สำหรับเก็บ Feature Map (ผลลัพธ์จาก Layer ตอน Forward Pass)
gradients = []    # สำหรับเก็บ Gradients (ค่าความชันที่ไหลย้อนกลับมาตอน Backward Pass)

def get_feature_maps_hook(module, input, output):
    """Hook ที่จะถูกเรียกใช้หลังจากการทำ Forward Pass ใน Layer ที่กำหนด เพื่อดักจับค่า output (Feature Map)"""
    feature_maps.append(output)

def get_gradients_hook(module, grad_in, grad_out):
    """Hook ที่จะถูกเรียกใช้ระหว่างการทำ Backward Pass เพื่อดักจับค่า gradient ที่ไหลผ่าน Layer"""
    gradients.append(grad_out[0])

class LLaVADentist:
    def __init__(self, base_model_path, adapter_path):
        # กำหนด Device ที่จะใช้ (GPU ถ้ามี, มิฉะนั้นใช้ CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model with PEFT on device: {self.device}")
        
        # --- การตั้งค่า Quantization (การบีบอัดโมเดล) ---
        # โหลดโมเดลแบบ 4-bit เพื่อประหยัดหน่วยความจำ GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # --- โหลดโมเดลพื้นฐาน (Base Model) ---
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto" # ให้ transformers จัดการการวางโมเดลบน device อัตโนมัติ
        )
        
        # --- โหลด Processor ---
        # Processor ทำหน้าที่เตรียมข้อมูล (รูปภาพและข้อความ) ให้พร้อมสำหรับโมเดル
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

        # --- ส่วนแก้ไขที่สำคัญ (Critical Fix) ---
        # ต้องทำก่อนที่จะโหลด PEFT adapter
        print("Applying critical fix to multi_modal_projector...")
        for name, param in base_model.named_parameters():
            if "multi_modal_projector" in name:
                # แปลง Data Type ของ multi_modal_projector กลับเป็น float32
                # เพื่อให้ PEFT สามารถตั้งค่าให้มัน trainable ได้โดยไม่เกิด error
                param.data = param.data.to(torch.float32)
        print("Fix applied.")

        # --- โหลดโมเดล PEFT (โหลด Adapter) ---
        # นำ LoRA adapter ที่เรา fine-tune ไว้มา "หุ้ม" โมเดลพื้นฐาน
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval() # ตั้งค่าโมเดลเป็น evaluation mode (ไม่ทำการ training)
        print("PEFT model initialized successfully.")

    def generate_narrative(self, image_path, instruction):
        try:
            # โหลดรูปภาพจาก path ที่กำหนด และแปลงเป็น RGB
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return f"Error: Image not found at {image_path}"
            
        # สร้าง Prompt ตาม format ของ LLaVA
        prompt = f"USER: <image>\n{instruction}\nASSANT:"
        # ใช้ Processor เพื่อแปลงรูปภาพและข้อความเป็น Tensor ที่โมเดลเข้าใจ
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)

        with torch.no_grad(): # ปิดการคำนวณ Gradient เพื่อประหยัดหน่วยความจำและเพิ่มความเร็ว
            # --- สร้างข้อความด้วยพารามิเตอร์การสุ่ม (Aggressive Sampling) ---
            # ใช้พารามิเตอร์เหล่านี้เพื่อ "กระตุ้น" ให้โมเดลสร้างคำตอบใหม่ๆ และไม่คัดลอก Prompt กลับมา
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256,         # จำกัดความยาวสูงสุดของคำตอบ
                do_sample=True,             # เปิดโหมดการสุ่ม (Sampling)
                temperature=0.7,            # เพิ่มความหลากหลายและความคิดสร้างสรรค์ของคำตอบ
                top_k=50,                   # จำกัดการสุ่มให้อยู่ในกลุ่มคำที่น่าจะเป็นที่สุด 50 คำแรก
                repetition_penalty=1.2      # ลดโอกาสที่โมเดลจะพูดคำซ้ำๆ
            )
        
        # --- ส่วนของการถอดรหัส (Decoding) และดีบัก ---
        # ถอดรหัสผลลัพธ์ดิบ (รวม special tokens) เพื่อใช้ในการดีบัก
        raw_generated_text = self.processor.decode(outputs[0], skip_special_tokens=False)
        print(f"DEBUG: Raw model output: '{raw_generated_text}'")
        
        # ถอดรหัสผลลัพธ์แบบสะอาด (ไม่รวม special tokens) เพื่อนำไปใช้งาน
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # --- Logic การแยกข้อความตอบกลับที่ทนทานขึ้น ---
        if "ASSISTANT:" in generated_text:
            # แยกเอาเฉพาะส่วนที่เป็นคำตอบของ Assistant
            response = generated_text.split("ASSISTANT:")[1].strip()
            return response if response else "Error: Model returned an empty response after 'ASSISTANT:'."
        else:
            # กรณีที่โมเดลไม่ได้สร้าง "ASSISTANT:" ออกมา ให้พยายามลบส่วน prompt ทิ้ง
            clean_text = generated_text.replace(f"USER: <image>\n{instruction}".strip(), "").strip()
            return clean_text if clean_text else "Error: Model produced no parsable output."

    def generate_xai_heatmap(self, image_path, full_text, keyword):
        global feature_maps, gradients # เรียกใช้ตัวแปร global
        
        # ตรวจสอบว่า Narrative ที่ได้มานั้นสมบูรณ์หรือไม่ ก่อนจะเริ่มทำ XAI
        if not full_text or "ASSISTANT:" not in full_text or not full_text.split("ASSISTANT:")[1].strip():
             return None
        try:
            # ระบุ Path ไปยัง Layer สุดท้ายของ Vision Tower ที่เราต้องการดักจับข้อมูล
            vision_tower_last_layer = self.model.base_model.model.vision_tower.vision_model.encoder.layers[-1].layer_norm2
        except AttributeError:
            return None
        
        # ติดตั้ง Hooks เข้ากับ Layer ที่ระบุ
        forward_hook = vision_tower_last_layer.register_forward_hook(get_feature_maps_hook)
        backward_hook = vision_tower_last_layer.register_full_backward_hook(get_gradients_hook)
        heatmap = None
        try:
            # (ส่วนที่เหลือของฟังก์ชันนี้คือกระบวนการคำนวณ Heatmap ที่คล้ายกับ Grad-CAM)
            image = Image.open(image_path).convert("RGB")
            # แปลง keyword ที่สนใจเป็น token ID
            keyword_token_ids = self.processor.tokenizer.encode(keyword, add_special_tokens=False)
            if not keyword_token_ids: return None
            target_token_id = keyword_token_ids[-1] # ใช้ token สุดท้ายถ้าคำนั้นมีหลาย token
            
            # เตรียมข้อมูลเข้าสำหรับโมเดล
            inputs = self.processor(text=full_text, images=image, return_tensors="pt").to(self.device)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
            
            # ล้างค่าเก่าใน global variables และ reset gradients
            feature_maps.clear(); gradients.clear(); self.model.zero_grad()
            
            # ทำ Forward Pass เพื่อให้ได้ logits และดักจับ Feature Map ผ่าน hook
            model_output = self.model(**inputs, output_hidden_states=True)
            logits = model_output.logits
            
            # เลือก logit ของ token เป้าหมาย (ที่ตำแหน่งเกือบท้ายสุด)
            target_logit = logits[0, -2, target_token_id]
            # ทำ Backward Pass จาก logit นั้น เพื่อคำนวณ Gradient และดักจับผ่าน hook
            target_logit.backward()
            
            # คำนวณ Heatmap จาก Feature Map และ Gradient ที่ดักจับมาได้
            if gradients and feature_maps:
                pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
                last_feature_maps = feature_maps[0].squeeze(0)
                for i in range(last_feature_maps.shape[0]): last_feature_maps[i, :, :] *= pooled_gradients[i]
                heatmap = torch.mean(last_feature_maps, dim=0).cpu().detach().numpy()
                heatmap = np.maximum(heatmap, 0)
                if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
        except Exception as e:
            print(f"An error occurred during heatmap generation for '{keyword}': {e}")
        finally:
            # นำ Hooks ออกเสมอ ไม่ว่าจะเกิด error หรือไม่ เพื่อป้องกัน memory leak
            forward_hook.remove(); backward_hook.remove()
        return heatmap
        
    def superimpose_heatmap(self, image_path, heatmap):
        # --- ฟังก์ชันสำหรับสร้างภาพ Visualization ---
        img = cv2.imread(image_path)
        # ปรับขนาด Heatmap ให้เท่ากับรูปภาพต้นฉบับ
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        # ใส่สีให้กับ Heatmap (JET colormap ให้สีรุ้ง)
        heatmap_colored = np.uint8(255 * heatmap_resized); heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        # นำ Heatmap มาวางซ้อนบนรูปภาพต้นฉบับ โดยปรับความโปร่งใส (alpha=0.5)
        superimposed_img = heatmap_colored * 0.5 + img; superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img