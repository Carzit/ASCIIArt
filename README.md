# ASCII Art Converter

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](pyproject.toml)

一个功能强大的Python工具，用于将图片和视频转换为ASCII艺术。支持多种颜色模式、字符集预设、图像增强选项，以及视频动画导出功能。

## ✨ 功能特性

### 🖼️ 图片转换
- **多种颜色模式**：无颜色、ANSI 16色、24位真彩色、HTML
- **丰富的字符集**：内置5种预设字符集，支持自定义字符集
- **图像增强**：Gamma矫正、对比度调整、反相处理
- **智能缩放**：支持按宽度、高度或比例缩放，自动计算最佳尺寸
- **字符比例补偿**：自动调整字符宽高比，确保输出图像不变形
- **抖动算法**：使用Bayer矩阵进行有序抖动，提升图像质量

### 🎬 视频转换
- **实时播放**：在终端中实时播放ASCII动画
- **多格式导出**：支持导出为MP4视频、GIF动画、文本帧序列
- **帧率控制**：可自定义播放和导出帧率
- **字体渲染**：支持自定义字体和颜色，生成高质量视频

### 🎨 输出格式
- **终端显示**：支持ANSI转义序列，兼容大多数终端
- **HTML输出**：生成带颜色的HTML文件，可在浏览器中查看
- **文件保存**：保存为纯文本或HTML格式
- **视频导出**：高质量MP4和GIF格式

## 🚀 快速开始

### 环境要求
- Python 3.13+
- 支持的操作系统：Windows、macOS、Linux

### 安装依赖

```bash
# 使用uv安装（推荐）
uv sync

# 或使用pip安装
pip install -r requirements.txt

# 注意：YAML配置功能需要PyYAML，已包含在依赖中
```

### 基本使用

#### 图片转ASCII
```bash
# 基本转换
python ascii_image.py -i image.jpg

# 使用配置文件
python ascii_image.py -i image.jpg -c config.yaml

# 指定输出尺寸和颜色
python ascii_image.py -i image.jpg --width 100 --color ansi

# 保存到文件
python ascii_image.py -i image.jpg -o output.txt --color html
```

#### 视频转ASCII
```bash
# 在终端播放
python ascii_video.py -i video.mp4

# 使用配置文件
python ascii_video.py -i video.mp4 -c config.yaml

# 导出为MP4
python ascii_video.py -i video.mp4 -m output.mp4

# 导出为GIF
python ascii_video.py -i video.mp4 -g output.gif
```

## 📖 详细使用说明

### 图片转换选项

#### 尺寸控制
```bash
--width 120          # 指定输出宽度（字符数）
--height 60          # 指定输出高度（字符数）
--scale 0.5          # 统一缩放比例
--char-aspect 0.45   # 字符宽高比补偿
```

#### 视觉效果
```bash
--charset standard    # 字符集：sparse|standard|dense|blocks|binary
--color none          # 颜色模式：none|ansi|truecolor|html
--bg black            # HTML背景色
--invert              # 反相处理
--gamma 1.2           # Gamma矫正
--contrast 1.5        # 对比度调整
--dither              # 启用抖动
```

#### 输出控制
```bash
-i, --input          # 输入图片路径（必需）
-o, --output         # 输出文件路径（可选，省略则打印到终端）
-c, --config         # YAML配置文件路径（可选）
```

### 视频转换选项

#### 基本参数
```bash
-i, --input          # 输入视频路径（必需）
-c, --config         # YAML配置文件路径（可选）
--fps 24             # 播放/导出帧率
--no-play            # 不在终端播放
```

#### 导出选项
```bash
-o, --output-dir     # 输出帧目录（每帧一个.txt文件）
-m, --save-mp4       # 导出为MP4文件
-g, --save-gif       # 导出为GIF文件
```

#### 渲染选项
```bash
--font font.ttf      # 字体文件路径
--font-size 14       # 字体大小
--fg #FFFFFF         # 前景色（十六进制）
--bg #000000         # 背景色（十六进制）
```

## ⚙️ 配置选项

项目支持通过`config.yaml`文件进行配置，支持所有命令行参数对应的配置项。

### 配置文件优先级
配置优先级从高到低：**命令行参数 > YAML配置文件 > 默认值**

### 基本配置示例

```yaml
# config.yaml
# 输出尺寸
width: 120           # 输出宽度（字符数）
height: 60           # 输出高度（字符数）
scale: 0.5           # 缩放比例

# 字符集和颜色
charset: "standard"  # 字符集
color: "none"        # 颜色模式
bg: "black"          # 背景色

# 图像处理
invert: false        # 是否反相
gamma: 1.0           # Gamma值
contrast: 1.0        # 对比度
char_aspect: 0.45    # 字符宽高比
dither: false        # 是否启用抖动
```

### 使用配置文件

```bash
# 图片转换
python ascii_image.py -i image.jpg -c config.yaml

# 视频转换
python ascii_video.py -i video.mp4 -c config.yaml

# 配置文件可以与命令行参数混合使用
python ascii_image.py -i image.jpg -c config.yaml --width 150 --color ansi
```

### 配置文件优势
- **批量处理**：为多张图片/视频设置统一的转换参数
- **团队协作**：团队成员可以共享相同的转换配置
- **参数管理**：避免重复输入长命令行参数
- **版本控制**：配置文件可以纳入版本控制，跟踪参数变化

## 🎯 字符集预设

项目内置了5种字符集预设：

- **`sparse`**: ` .:-=+*#%@` - 稀疏字符集，适合简单图像
- **`standard`**: ` .'"^,,:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$` - 标准字符集，平衡效果和细节
- **`blocks`**: ` ▁▂▃▄▅▆▇█` - 块状字符集，现代终端友好
- **`binary`**: ` .` - 二进制字符集，极简风格

## 🌈 颜色模式说明

### 无颜色模式 (`none`)
纯文本输出，适合所有终端和文件格式。

### ANSI 16色模式 (`ansi`)
使用标准ANSI 16色调色板，兼容性最好。

### 24位真彩色模式 (`truecolor`)
支持1677万种颜色，需要支持真彩色的终端。

### HTML模式 (`html`)
生成带颜色的HTML文件，使用`<span>`标签着色。

## 📁 项目结构

```
ASCIIArt/
├── ascii_image.py      # 图片转ASCII核心模块
├── ascii_video.py      # 视频转ASCII模块
├── config.yaml         # 默认配置文件示例
├── pyproject.toml      # 项目配置
├── uv.lock            # 依赖锁定文件
├── test.jpg           # 测试图片
└── README.md          # 项目说明文档
```

## 🔧 开发环境

### 依赖管理
项目使用`uv`进行依赖管理，确保环境一致性：

```bash
# 安装uv（如果未安装）
pip install uv

# 同步依赖
uv sync

# 激活虚拟环境
uv shell
```

### 代码质量
- 使用类型注解（Python 3.13+）
- 遵循PEP 8代码风格
- 完整的文档字符串

## 📝 使用示例

### 示例1：基本图片转换
```bash
# 将图片转换为80字符宽的ASCII艺术
python ascii_image.py -i photo.jpg --width 80 --color ansi

# 使用配置文件进行转换
python ascii_image.py -i photo.jpg -c config.yaml
```

### 示例2：高质量HTML输出
```bash
# 生成带颜色的HTML文件
python ascii_image.py -i photo.jpg -o art.html --color html --width 120
```

### 示例3：视频动画
```bash
# 在终端播放视频动画
python ascii_video.py -i animation.mp4 --width 100 --color truecolor

# 导出为GIF动画
python ascii_video.py -i animation.mp4 -g output.gif --width 80
```

### 示例4：批量处理
```bash
# 处理多张图片
for img in *.jpg; do
    python ascii_image.py -i "$img" -o "${img%.jpg}.txt" --width 100
done

# 使用配置文件批量处理
for img in *.jpg; do
    python ascii_image.py -i "$img" -o "${img%.jpg}.txt" -c config.yaml
done
```

### 示例5：配置文件管理
```bash
# 创建不同风格的配置文件
# config_high_quality.yaml
width: 150
charset: "dense"
color: "truecolor"
gamma: 1.2
contrast: 1.1

# config_terminal.yaml
width: 80
charset: "standard"
color: "ansi"
char_aspect: 0.5

# 使用不同配置转换
python ascii_image.py -i photo.jpg -c config_high_quality.yaml -o high_quality.txt
python ascii_image.py -i photo.jpg -c config_terminal.yaml -o terminal.txt
```

