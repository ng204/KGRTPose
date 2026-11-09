#!/usr/bin/env python3
"""
生成可视化结果的HTML索引页面
"""

import os
from pathlib import Path

OUTPUT_ROOT = "/home/ng204/GH/ultralytics+rtmo/runs/keshihua/2"
ANIMALPOSE_DIR = f"{OUTPUT_ROOT}/animalpose_yolo"
ATRW_DIR = f"{OUTPUT_ROOT}/atrw"


def get_image_list(directory):
    """获取目录中的所有图片"""
    images = []
    for img_file in sorted(os.listdir(directory)):
        if img_file.endswith(('.jpg', '.png')):
            images.append({
                'name': img_file,
                'path': f"{os.path.basename(directory)}/{img_file}"
            })
    return images


def generate_html():
    """生成HTML索引页面"""
    animalpose_images = get_image_list(ANIMALPOSE_DIR)
    atrw_images = get_image_list(ATRW_DIR)

    # 生成 AnimalPose 图片列表 JavaScript
    animalpose_js = "[\n"
    for img in animalpose_images:
        animalpose_js += f"    {{'name': '{img['name']}', 'path': '{img['path']}'}},\n"
    animalpose_js += "]"

    # 生成 ATRW 图片列表 JavaScript
    atrw_js = "[\n"
    for img in atrw_images:
        atrw_js += f"    {{'name': '{img['name']}', 'path': '{img['path']}'}},\n"
    atrw_js += "]"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据集可视化预览</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .dataset-section {{
            margin-bottom: 50px;
        }}

        .dataset-title {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .dataset-title h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 5px;
        }}

        .dataset-title p {{
            color: #666;
            font-size: 0.95em;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}

        .image-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }}

        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }}

        .image-container {{
            overflow: hidden;
            background: #f0f0f0;
            height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .image-info {{
            padding: 15px;
        }}

        .image-name {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
            word-break: break-all;
            font-size: 0.9em;
        }}

        .image-type {{
            display: inline-block;
            padding: 3px 8px;
            background: #667eea;
            color: white;
            border-radius: 4px;
            font-size: 0.8em;
        }}

        .stats {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .stats h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}

        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            padding: 20px;
        }}

        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .legend h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .legend-content {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            font-size: 0.9em;
            line-height: 1.6;
        }}

        .legend-item {{
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎨 数据集可视化预览</h1>
            <p>关键点检测数据集 - AnimalPose YOLO & ATRW</p>
        </header>

        <!-- AnimalPose YOLO 数据集 -->
        <div class="dataset-section">
            <div class="dataset-title">
                <h2>🐟 AnimalPose YOLO 数据集</h2>
                <p>鱼类关键点检测 - 6个关键点</p>
            </div>

            <div class="legend">
                <h3>关键点说明</h3>
                <div class="legend-content">
                    <div class="legend-item">
                        <strong>关键点0:</strong> mouth (嘴部)
                    </div>
                    <div class="legend-item">
                        <strong>关键点1:</strong> eye_left (左眼)
                    </div>
                    <div class="legend-item">
                        <strong>关键点2:</strong> eye_right (右眼)
                    </div>
                    <div class="legend-item">
                        <strong>关键点3:</strong> dorsal_mid (背鳍中点)
                    </div>
                    <div class="legend-item">
                        <strong>关键点4:</strong> spine_mid (脊椎中点)
                    </div>
                    <div class="legend-item">
                        <strong>关键点5:</strong> caudal_tip (尾部)
                    </div>
                </div>
            </div>

            <div class="gallery" id="animalpose-gallery">
                <!-- 动态生成 -->
            </div>
        </div>

        <!-- ATRW 数据集 -->
        <div class="dataset-section">
            <div class="dataset-title">
                <h2>🐯 ATRW 数据集</h2>
                <p>老虎关键点检测 - 17个关键点（COCO格式）</p>
            </div>

            <div class="legend">
                <h3>关键点说明</h3>
                <div class="legend-content">
                    <div class="legend-item">
                        使用 COCO Pose 标准关键点格式（17个关键点）
                    </div>
                    <div class="legend-item">
                        包括: 鼻子、眼睛、耳朵、肩膀、肘部、腕部、髋部、膝部、踝部等
                    </div>
                </div>
            </div>

            <div class="gallery" id="atrw-gallery">
                <!-- 动态生成 -->
            </div>
        </div>

        <!-- 统计信息 -->
        <div class="stats">
            <h3>📊 可视化统计</h3>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">30</div>
                    <div class="stat-label">AnimalPose 图片</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">30</div>
                    <div class="stat-label">ATRW 图片</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">60</div>
                    <div class="stat-label">总计图片数</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">6.6 MB</div>
                    <div class="stat-label">总大小</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>✅ 可视化完成 | 所有图片已保存在 /home/ng204/GH/ultralytics+rtmo/runs/keshihua/2</p>
        </div>
    </div>

    <script>
        // 生成图片卡片
        function generateGallery(containerId, images) {{
            const container = document.getElementById(containerId);

            images.forEach((img, index) => {{
                const card = document.createElement('div');
                card.className = 'image-card';
                card.innerHTML = `
                    <div class="image-container">
                        <img src="${{img.path}}" alt="${{img.name}}" loading="lazy">
                    </div>
                    <div class="image-info">
                        <div class="image-name">${{img.name}}</div>
                        <span class="image-type">${{index + 1}}/30</span>
                    </div>
                `;

                // 添加点击查看大图功能
                card.addEventListener('click', () => {{
                    openImageViewer(img.path, img.name);
                }});

                container.appendChild(card);
            }});
        }}

        // 打开图片查看器
        function openImageViewer(imagePath, imageName) {{
            window.open(imagePath, '_blank');
        }}

        // 初始化图片列表
        const animalposImages = {animalpose_js};
        const atrwImages = {atrw_js};

        // 生成图片库
        generateGallery('animalpose-gallery', animalposImages);
        generateGallery('atrw-gallery', atrwImages);
    </script>
</body>
</html>
"""

    # 保存HTML文件
    html_path = f"{OUTPUT_ROOT}/index.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ HTML索引页面已生成: {html_path}")


if __name__ == "__main__":
    generate_html()
