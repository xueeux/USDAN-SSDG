import xml.etree.ElementTree as ET

def add_axes_labels(svg_path, output_path, x_label="iter", y_label="consistency_loss"):
    """
    向已有的 SVG 图表添加横纵坐标轴文字
    """
    # 注册默认命名空间，避免输出带有 ns0 前缀
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 根据 viewBox 和图中实际轴线位置估算标签坐标
    # viewBox: 0 0 1251.666748046875 400
    # 左侧 Y 轴位置 x ≈ 50，右侧 X 轴最远 x ≈ 1250，底部 Y ≈ 377
    # 横轴标签放在 X 轴中点下方，纵轴标签放在 Y 轴中点左侧（旋转 -90°）
    x_center = 650      # (50 + 1250) / 2
    x_label_y = 420     # 底部刻度区域以下，原图最高 y=400，取 390 留白

    y_center = 178      # (5 + 350) / 2
    y_label_x = 15      # 左侧数字最左边约 40，取 30 更靠左，避免重叠

    # 设置文本样式（与原图数字风格保持一致）
    text_style = (
        "font-family: Roboto, sans-serif; "
        "font-size: 16px; "
        "fill: #212121; "
        "text-anchor: middle;"
    )

    # 1. 添加横坐标标签
    x_elem = ET.Element("text", {
        "x": str(x_center),
        "y": str(x_label_y),
        "style": text_style
    })
    x_elem.text = x_label
    root.append(x_elem)

    # 2. 添加纵坐标标签（旋转 -90°，旋转中心为该点）
    y_elem = ET.Element("text", {
        "x": str(y_label_x),
        "y": str(y_center),
        "transform": f"rotate(-90, {y_label_x}, {y_center})",
        "style": text_style
    })
    y_elem.text = y_label
    root.append(y_elem)

    # 保存文件（保留原 XML 声明和编码）
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ 已生成带坐标标签的 SVG：{output_path}")

if __name__ == "__main__":
    # 请根据你电脑上的实际路径修改
    input_svg = r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\train_consistency_loss (1).svg"
    output_svg = r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\save_results\train_consistency_loss_labeled.svg"

    add_axes_labels(input_svg, output_svg)