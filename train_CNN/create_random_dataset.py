#!/usr/bin/env python
"""
生成随机图片数据集
支持指定图片尺寸、数量和目录数量
"""
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def generate_random_image(size, format='JPEG'):
    """
    生成一张随机图片
    
    Args:
        size: 图片尺寸 (width, height)
        format: 图片格式，默认 'JPEG'
    
    Returns:
        PIL Image 对象
    """
    # 生成随机 RGB 图片
    width, height = size
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_array, 'RGB')
    return image


def create_random_dataset(output_dir, num_images, image_size, num_dirs, 
                         image_format='JPEG', prefix='img'):
    """
    创建随机图片数据集
    
    Args:
        output_dir: 输出根目录
        num_images: 图片总数量
        image_size: 图片尺寸 (width, height)
        num_dirs: 目录数量
        image_format: 图片格式，默认 'JPEG'
        prefix: 图片文件名前缀，默认 'img'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 计算每个目录的图片数量
    images_per_dir = num_images // num_dirs
    remainder = num_images % num_dirs
    
    # 确定文件扩展名
    ext_map = {'JPEG': '.jpg', 'PNG': '.png', 'BMP': '.bmp'}
    ext = ext_map.get(image_format, '.jpg')
    
    image_idx = 0
    
    # 为每个目录生成图片
    for dir_idx in range(num_dirs):
        # 创建子目录
        subdir = output_path / f"dir_{dir_idx:04d}"
        subdir.mkdir(parents=True, exist_ok=True)
        
        # 计算当前目录的图片数量（余数分配给前几个目录）
        current_dir_images = images_per_dir + (1 if dir_idx < remainder else 0)
        
        # 生成图片
        for local_idx in tqdm(range(current_dir_images), 
                             desc=f"生成目录 {dir_idx+1}/{num_dirs}"):
            # 生成随机图片
            image = generate_random_image(image_size, image_format)
            
            # 保存图片
            filename = f"{prefix}_{image_idx:08d}{ext}"
            filepath = subdir / filename
            image.save(filepath, format=image_format, quality=95)
            
            image_idx += 1
    
    print(f"✅ 成功生成 {num_images} 张图片")
    print(f"   输出目录: {output_dir}")
    print(f"   图片尺寸: {image_size[0]}x{image_size[1]}")
    print(f"   目录数量: {num_dirs}")
    print(f"   图片格式: {image_format}")


def main():
    parser = argparse.ArgumentParser(description='生成随机图片数据集')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出根目录路径')
    parser.add_argument('--num-images', type=int, required=True,
                       help='图片总数量')
    parser.add_argument('--image-size', type=int, nargs=2, required=True,
                       metavar=('WIDTH', 'HEIGHT'),
                       help='图片尺寸 (宽度 高度)')
    parser.add_argument('--num-dirs', type=int, required=True,
                       help='目录数量')
    parser.add_argument('--format', type=str, default='JPEG',
                       choices=['JPEG', 'PNG', 'BMP'],
                       help='图片格式 (默认: JPEG)')
    parser.add_argument('--prefix', type=str, default='img',
                       help='图片文件名前缀 (默认: img)')
    
    args = parser.parse_args()
    
    create_random_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=tuple(args.image_size),
        num_dirs=args.num_dirs,
        image_format=args.format,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()

