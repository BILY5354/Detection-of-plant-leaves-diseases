#2022.4.7修改，增加了格式转换
#2022.5.3修改，增加了背景图，优化了一些配置

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ssd import SSD
import base64
import global_num

crop   =   False

st.header("基于OpenCV与卷积神经网络的植物叶片病害实时检测与识别系统")
st.write("选择你需要识别的叶片图片:")
with st.expander("使用说明"):
    st.write("""
            - 输入图片
            - 先目标检测
            - 再分类识别
        """)

#只能输入jpg文件
uploaded_file = st.file_uploader("可以任意输入以下格式图片",type=["jpg", "png", "bmp", "jpeg","webp","psd","dxf","gif","apng","tif","pcx","tga","exif","fpx","svg","cdr","pcd","ufo","eps","ai","raw","WMF","avif"])

#uploaded_file = st.file_uploader("Choose an image...")


if uploaded_file is not None:

    ssd = SSD()
    #image = Image.open(uploaded_file)

    ####转成.JPG文件###
    #将输入的文件使用PIL打开给img，img以“image.jpg”保存，为了转变为.jpg格式
    img = Image.open(uploaded_file).convert('RGB')
    img.save('image.jpg',quality=95)
    #PIL打开转换后保存的.jpg图片，赋值给image，完成格式的转换
    image = Image.open('image.jpg')



    #st.image(image, caption='输入图片',width=30, use_column_width=True)
    st.image(image, caption='输入图片显示',use_column_width=True)

    st.header('输出叶片检测结果：')
    r_image = ssd.detect_image(image)
    st.image(r_image, caption='Output Image', use_column_width=True)


    if global_num.thing > 0:
        st.header('对识别叶片分别进行病害检测：')
        ssd.detect_image2(image)

st.header('                                                                                                                                                                                                                                                                                                           ')
st.header('                                                                                                                                                                                                                                                                                                           ')
st.header('                                                                                                                                                                                                                                                                                                           ')
st.header('                                                                                                                                                                                                                                                                                                           ')
st.write(
        """
        
        
        
        - 五邑大学智能制造学部200735徐泽华毕业设计wed端💬
        - QQ邮箱：1290820085@qq.com✨
        """
    )


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
   with open(bin_file, 'rb') as f:
       data = f.read()
   return base64.b64encode(data).decode()
def set_png_as_page_bg(png_file):
   bin_str = get_base64_of_bin_file(png_file)
   page_bg_img = '''
<style>
.stApp {
 background-image: url("data:image/png;base64,%s");
 background-size: cover;
}
</style>
   ''' % bin_str
   st.markdown(page_bg_img, unsafe_allow_html=True)
   return
set_png_as_page_bg('1.png')