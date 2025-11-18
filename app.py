import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
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
        """加载PMC_PhaseNet模型"""
        try:
            self.model = YOLO(self.model_path)
            st.sidebar.success("✅ PMC_PhaseNet模型加载成功")
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            self.model = None
    
    def detect_image(self, image):
        """执行图像检测"""
        try:
            if self.model is not None:
                # 使用PMC_PhaseNet模型进行推理
                results = self.model(image)
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls.item())
                            confidence = box.conf.item()
                            bbox = box.xyxy[0].tolist()
                            
                            class_info = PINE_FLOWER_CLASSES.get(class_id, {
                                'name': 'unknown', 'color': (255, 165, 0), 'display_name': 'Unknown Stage'
                            })
                            
                            detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'class_name': class_info['name'],
                                'display_name': class_info['display_name'],
                                'class_id': class_id,
                                'color': class_info['color']
                            })
                return detections, self.draw_detections(image, detections)
            else:
                # 模拟检测用于测试
                return self.mock_detect(image), self.draw_detections(image, self.mock_detect(image))
                
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
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            display_name = det['display_name']
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签背景
            label = f"{display_name} {conf:.2f}"
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(label) * 10
                text_height = 20
                
            draw.rectangle([x1, max(y1 - text_height - 10, 0), 
                          x1 + text_width + 10, y1], fill=color)
            
            # 绘制文本
            text_y = max(y1 - text_height - 5, 5)
            draw.text((x1 + 5, text_y), label, fill=(255, 255, 255), font=font)
        
        return pil_image
    
    def mock_detect(self, image):
        """模拟检测用于测试"""
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
            confidence = round(0.6 + random.random() * 0.3, 2)
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
    # 标题和介绍
    st.title("🌲 松花物候期识别系统")
    st.markdown("基于PMC_PhaseNet - 检测伸长期、成熟期、衰退期")
    
    # 侧边栏信息
    st.sidebar.title("关于系统")
    st.sidebar.info("""
    本系统使用PMC_PhaseNet深度学习模型对油松雄球花的物候期进行自动识别和分类。
    
    **支持的物候期:**
    - 🌱 伸长期 (Elongation Stage)
    - 🍎 成熟期 (Ripening Stage) 
    - 🍂 衰退期 (Decline Stage)
    """)
    
    # 文件上传组件[citation:1]
    uploaded_file = st.file_uploader(
        "选择图像文件",
        type=['png', 'jpg', 'jpeg'],
        help="支持格式: JPG, PNG, JPEG"
    )
    
    if uploaded_file is not None:
        # 显示文件信息[citation:1]
        file_details = {
            "文件名": uploaded_file.name,
            "文件大小": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "文件类型": uploaded_file.type
        }
        st.write("文件详情:", file_details)
        
        # 加载检测器
        detector = load_detector()
        
        # 开始检测按钮[citation:1]
        if st.button("开始检测", type="primary"):
            with st.spinner("检测中..."):
                try:
                    # 使用PIL加载图像[citation:1]
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # 显示原图和处理结果
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("原始图像")
                        st.image(image, use_container_width=True)
                    
                    # 执行检测
                    detections, result_image = detector.detect_image(image)
                    
                    with col2:
                        st.subheader("检测结果")
                        st.image(result_image, use_container_width=True)
                    
                    # 显示统计信息
                    st.subheader("📊 检测统计")
                    stats = detector.get_statistics(detections)
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.metric("总检测数", stats['total_count'])
                    
                    with stat_col2:
                        stages_count = len(stats['by_stage'])
                        st.metric("物候期类型", stages_count)
                    
                    with stat_col3:
                        if detections:
                            avg_confidence = np.mean([d['confidence'] for d in detections])
                            st.metric("平均置信度", f"{avg_confidence:.2f}")
                    
                    # 详细检测结果
                    st.subheader("🔍 检测详情")
                    if detections:
                        for i, det in enumerate(detections):
                            st.write(
                                f"**松花 {i+1}**: {det['display_name']} "
                                f"(置信度: {det['confidence']:.2f})"
                            )
                    else:
                        st.info("未检测到松花")
                    
                    # 检测完成时间
                    st.success(f"检测完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"图像处理错误: {e}")

if __name__ == "__main__":
    main()
