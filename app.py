import streamlit as st
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
import torch
from PIL import Image, ImageDraw, ImageFont

# 页面配置
st.set_page_config(
    page_title="Pine Flower Phenology Recognition",
    page_icon="🌲",
    layout="wide"
)

# 模型文件检查
model_path = 'models/best.pt'
if os.path.exists(model_path):
    st.sidebar.success(f"✅ 模型文件加载成功 ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
else:
    st.sidebar.error("❌ 模型文件未找到")

# 松花物候期类别映射
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'display_name': 'Elongation Stage'},
    1: {'name': 'ripening stage', 'color': (255, 165, 0), 'display_name': 'Ripening Stage'},
    2: {'name': 'decline stage', 'color': (255, 0, 0), 'display_name': 'Decline Stage'}
}

class PineFlowerDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """直接使用PyTorch加载模型"""
        try:
            self.model = torch.load(self.model_path, map_location='cpu')
            if hasattr(self.model, 'eval'):
                self.model.eval()
            st.sidebar.success("✅ PMC_PhaseNet模型加载成功")
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            st.info("使用模拟检测模式")
            self.model = None
    
    def detect_image(self, image):
        """执行图像检测"""
        try:
            if self.model is not None:
                # 这里简化处理，实际需要根据模型结构进行推理
                # 由于模型结构复杂，我们先使用模拟检测
                detections = self.mock_detect(image)
            else:
                detections = self.mock_detect(image)
                
            return detections, self.draw_detections(image, detections)
                
        except Exception as e:
            st.error(f"检测过程中出错: {e}")
            return [], image
    
    def draw_detections(self, image, detections):
        """在图像上绘制检测框"""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image.copy()
            
        draw = ImageDraw.Draw(pil_image)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            display_name = det['display_name']
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            label = f"{display_name} {conf:.2f}"
            text_width = len(label) * 10
            text_height = 20
                
            draw.rectangle([x1, max(y1 - text_height - 10, 0), 
                          x1 + text_width + 10, y1], fill=color)
            
            text_y = max(y1 - text_height - 5, 5)
            draw.text((x1 + 5, text_y), label, fill=(255, 255, 255))
        
        return pil_image
    
    def mock_detect(self, image):
        """模拟检测用于演示"""
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
            
        detections = []
        import random
        num_detections = random.randint(1, 3)
        
        for i in range(num_detections):
            x1 = random.randint(50, width - 200)
            y1 = random.randint(50, height - 200)
            x2 = x1 + random.randint(100, 300)
            y2 = y1 + random.randint(100, 300)
            confidence = round(0.7 + random.random() * 0.2, 2)
            class_id = random.randint(0, 2)
            class_info = PINE_FLOWER_CLASSES[class_id]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_info['name'],
                'display_name': class_info['display_name'],
                'class_id': class_id,
                'color': class_info['color']
            })
        return detections
    
    def get_statistics(self, detections):
        """获取检测统计信息"""
        stats = {'total_count': 0, 'by_stage': defaultdict(int)}
        if not detections:
            return stats
        
        stats['total_count'] = len(detections)
        for det in detections:
            stage = det['display_name']
            stats['by_stage'][stage] += 1
        
        return stats

# 初始化检测器
@st.cache_resource
def load_detector():
    return PineFlowerDetector('models/best.pt')

def main():
    st.title("🌲 松花物候期识别系统")
    st.markdown("基于PMC_PhaseNet - 检测伸长期、成熟期、衰退期")
    
    st.sidebar.info("""
    **系统说明:**
    - 当前使用模拟检测模式
    - 完整模型推理功能开发中
    - 支持三种物候期识别
    """)
    
    uploaded_file = st.file_uploader(
        "选择图像文件",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        file_details = {
            "文件名": uploaded_file.name,
            "文件大小": f"{uploaded_file.size / 1024 / 1024:.2f} MB"
        }
        st.write("文件详情:", file_details)
        
        detector = load_detector()
        
        if st.button("开始检测", type="primary"):
            with st.spinner("检测中..."):
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("原始图像")
                        st.image(image, use_container_width=True)
                    
                    detections, result_image = detector.detect_image(image)
                    
                    with col2:
                        st.subheader("检测结果")
                        st.image(result_image, use_container_width=True)
                    
                    # 显示统计信息
                    st.subheader("📊 检测统计")
                    stats = detector.get_statistics(detections)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总检测数", stats['total_count'])
                    with col2:
                        for stage, count in stats['by_stage'].items():
                            st.metric(f"{stage}", count)
                    
                    st.success(f"检测完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"处理错误: {e}")

if __name__ == "__main__":
    main()
