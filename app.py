import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="松花物候期识别系统",
    page_icon="🌲", 
    layout="wide"
)

st.title("🌲 松花物候期识别系统")
st.markdown("基于PMC_PhaseNet - 检测伸长期、成熟期、衰退期")

# 文件上传
uploaded_file = st.file_uploader("上传松花图像", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 显示图像
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    # 模拟检测结果
    st.success("✅ 系统运行正常！")
    st.info("""
    **模拟检测结果：**
    - 检测到 2 个松花雄球花
    - 物候期：成熟期 (Ripening Stage)
    - 平均置信度：0.85
    """)
    
    st.write("---")
    st.write("**技术说明：** 完整模型推理功能将在后续版本中集成")

st.sidebar.info("""
**系统信息：**
- 版本：1.0
- 状态：基础功能正常运行
- 下一步：集成PMC_PhaseNet模型
""")
