from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
# def add_legend_to_image(
#     image_path: str,
#     legend_items: list,          # 例如 [("ResNet50", (255,0,0)), ("ViT", (0,0,255))]
#     output_path: str = None,
#     position: str = "upper right",  # 目前仅实现了 'upper right'
#     font_path: str = None,           # 如无中文可留空，有中文需指定中文字体路径
#     font_size: int = 20,
#     box_alpha: int = 200,            # 0~255，背景透明度
#     text_color: tuple = (0,0,0),     # 文字颜色 (R,G,B)
#     box_color: tuple = (255,255,255),# 背景色 (R,G,B)
#     padding: int = 10,               # 内边距
#     marker_size: int = 16,           # 颜色方块大小
#     margin: int = 20                 # 图例距离图片边缘的距离
# ):
#     """
#     在图片右上角添加图例
#     legend_items:  [(名称, (R,G,B)), ...]
#     """
#     if output_path is None:
#         name, ext = os.path.splitext(image_path)
#         output_path = f"{name}_with_legend{ext}"
    
#     # 1. 打开图片
#     img = Image.open(image_path).convert("RGBA")
#     img_width, img_height = img.size
    
#     # 2. 加载字体
#     try:
#         if font_path and os.path.exists(font_path):
#             font = ImageFont.truetype(font_path, font_size)
#         else:
#             # 尝试获取默认字体
#             font = ImageFont.load_default()
#             # 如果默认字体太小，尝试放大（仅限默认字体，PIL 限制较多）
#             # 这里简单提示：默认字体不支持中文
#             print("警告：未指定中文字体，中文可能显示为方框。如需中文，请提供中文字体文件路径")
#     except:
#         font = ImageFont.load_default()
    
#     # 3. 计算图例尺寸
#     max_text_width = 0
#     for label, _ in legend_items:
#         # 获取文本大致宽度（Pillow 8.0+ 推荐用 getbbox）
#         bbox = font.getbbox(label)
#         text_width = bbox[2] - bbox[0]
#         if text_width > max_text_width:
#             max_text_width = text_width
    
#     legend_width = max_text_width + marker_size + 3 * padding
#     # 高度 = 条目数 * (font_size + 间距) + 上下padding
#     item_height = font_size + 8
#     legend_height = len(legend_items) * item_height + 2 * padding
    
#     # 4. 计算位置 (右上角)
#     x = img_width - margin - legend_width
#     y = margin
    
#     # 5. 创建一个临时图层用于绘制半透明背景和内容
#     overlay = Image.new("RGBA", img.size, (0,0,0,0))
#     draw = ImageDraw.Draw(overlay)
    
#     # 绘制背景矩形 (半透明)
#     draw.rectangle(
#         [x, y, x + legend_width, y + legend_height],
#         fill=(box_color[0], box_color[1], box_color[2], box_alpha),
#         outline=(0,0,0,100)
#     )
    
#     # 绘制每个条目
#     for idx, (label, color) in enumerate(legend_items):
#         # 条目垂直起始位置
#         item_y = y + padding + idx * item_height
#         # 绘制颜色方块
#         marker_x = x + padding
#         marker_y = item_y + (item_height - marker_size) // 2
#         draw.rectangle(
#             [marker_x, marker_y, marker_x + marker_size, marker_y + marker_size],
#             fill=color[:3] if len(color)==3 else color,
#             outline=(0,0,0)
#         )
#         # 绘制文本
#         text_x = marker_x + marker_size + padding
#         # 让文字垂直居中于条目区域
#         text_y = item_y + (item_height - font_size) // 2 - 2  # 微调偏移
#         draw.text((text_x, text_y), label, fill=text_color, font=font)
    
#     # 6. 合成并保存
#     combined = Image.alpha_composite(img, overlay)
#     combined = combined.convert("RGB")  # 转回RGB保存
#     combined.save(output_path)
#     print(f"图例已添加，保存至：{output_path}")
#     return output_path
# if __name__ == "__main__":
#     # 请替换为你的截图路径
#     screenshot_path = r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\entropy_loss.png"
    
#     # 定义模型名称和对应的曲线颜色（RGB元组）
#     # 你需要根据截图中的实际曲线颜色来填写RGB值
#     legend_data = [
#         ("entropy=0.05", (234,57,120)),    
#         ("entropy=0.1", (38,157,142)),     
#         ("entropy=0.15", (187,187,187)),  
#     ]
    
#     # 如果需要显示中文，请提供支持中文的字体文件路径（如微软雅黑、思源黑体等）
#     # Windows 示例：font_path = "C:/Windows/Fonts/msyh.ttc"
#     # macOS 示例：font_path = "/System/Library/Fonts/PingFang.ttc"
#     font_path = None   # 无中文可留空
    
#     add_legend_to_image(
#         image_path=screenshot_path,
#         legend_items=legend_data,
#         font_path=font_path,
#         font_size=22,
#         box_alpha=200,
#         text_color=(0,0,0),
#         box_color=(255,255,255)
#     )

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ================= 用户配置区 =================
image_paths = [
    r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\USDAN\ACC_valid.png",
    r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\USDAN\AUC_valid.png",
    r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\USDAN\HTER_valid.png",
    r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\USDAN\acc_train.png",
    r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\USDAN\loss_train.png",
]

legend_data = [
    ("OULU", (238,97,148)),
    ("OULU+AI_I",  (123,203,243)),
    ("OULU+AI_II", (189,189,189)),
]

output_path = r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\USDAN\merged_figures_with_legend.png"
# ==========================================

def rgb_to_normalized(rgb):
    return tuple(c / 255.0 for c in rgb)

legend_handles = []
for label, rgb in legend_data:
    normalized_color = rgb_to_normalized(rgb)
    patch = mpatches.Patch(color=normalized_color, label=label)
    legend_handles.append(patch)

# ---------- 关键修改 1：调整画布大小，让整体更加竖长 ----------
# 原 figsize=(15,6) 横向太长。根据每个子图希望的高/宽比～2，设置 figsize=(8,12)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))   # 宽度8英寸，高度12英寸 → 每个子图单元宽2.67英寸、高6英寸
plt.subplots_adjust(wspace=0, hspace=0.2)           # 完全消除子图间间隙
axes_flat = axes.flatten()

# 放置前5张图片
for i, img_path in enumerate(image_paths):
    try:
        img = Image.open(img_path).convert('RGB')
        # width, height = img.size
        # crop_width = int(width * (14/15))   # 保留左边 12/13
        # img_cropped = img.crop((0, 0, crop_width, height))
        ax = axes_flat[i]
        ax.imshow(img, aspect='auto')              # 关键修改 2：强制拉伸图像填充子图区域
        ax.axis('off')
        
        # 关键修改 3：设置子图显示框的高宽比，实现“纵向拉长”
        # set_box_aspect(ratio) 中 ratio = height/width，这里设为 2.5 使高度明显大于宽度
        #ax.set_box_aspect(2.5)                     # 数值越大，子图越“瘦高”
        
        # 添加 "iter" 文本：y 使用负值使文字位于图像外侧下方，但因 hspace=0，实际会与下一行紧贴？
        # 为了不覆盖下一行图像，将 y 设为 -0.08（子图坐标系内），同时调节垂直对齐方式
        ax.text(0.5, 0, 'iter',
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=10)
    except FileNotFoundError:
        print(f"错误：找不到图片 {img_path}")
        axes_flat[i].axis('off')
        axes_flat[i].text(0.5, 0.5, "Missing", ha='center', va='center')

# 第6个格子：图例，同样应用 box_aspect 保持风格统一
ax_legend = axes_flat[5]
ax_legend.axis('off')
ax_legend.set_box_aspect(2.5)                     # 与图像子图保持相同高宽比
ax_legend.legend(handles=legend_handles,
                 loc='center',
                 fontsize=12,
                 title="Parameter",
                 title_fontsize=14,
                 frameon=True,
                 fancybox=True,
                 shadow=True)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"完成！合并图片已保存至：{output_path}")