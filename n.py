"""
AvaMERG 数据集预处理脚本
功能：过滤掉所有缺失音频或视频的条目，生成干净的JSON文件
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('data_filter.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def filter_missing_files(
    json_path,
    video_dir, 
    audio_dir,
    output_json_path
):
    """
    过滤掉缺失文件的条目
    
    参数：
        json_path: 原始JSON文件路径
        video_dir: 视频文件夹路径
        audio_dir: 音频文件夹路径
        output_json_path: 输出的过滤后JSON文件路径
    """
    
    # 读取原始JSON
    logger.info(f"正在读取JSON文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同的JSON格式
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and 'data' in data:
        entries = data['data']
    else:
        entries = [data]
    
    logger.info(f"原始数据条目数: {len(entries)}")
    
    # 转换为Path对象
    video_dir = Path(video_dir)
    audio_dir = Path(audio_dir)
    
    # 过滤有效条目
    valid_entries = []
    missing_video_count = 0
    missing_audio_count = 0
    missing_both_count = 0
    
    for entry in tqdm(entries, desc="检查文件完整性"):
        # 获取文件ID（根据你的JSON结构调整）
        # 尝试不同的字段名
        file_id = None
        for key in ['id', 'video_id', 'audio_id', 'file_id', 'name']:
            if key in entry:
                file_id = str(entry[key])
                break
        
        if not file_id:
            logger.warning(f"无法找到ID字段: {entry}")
            continue
        
        # 检查视频文件是否存在
        video_exists = False
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            video_path = video_dir / f"{file_id}{ext}"
            if video_path.exists():
                video_exists = True
                entry['video_file'] = f"{file_id}{ext}"  # 记录实际文件名
                break
        
        # 如果没有扩展名的文件也检查
        if not video_exists:
            video_path = video_dir / file_id
            if video_path.exists():
                video_exists = True
                entry['video_file'] = file_id
        
        # 检查音频文件是否存在
        audio_exists = False
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        for ext in audio_extensions:
            audio_path = audio_dir / f"{file_id}{ext}"
            if audio_path.exists():
                audio_exists = True
                entry['audio_file'] = f"{file_id}{ext}"  # 记录实际文件名
                break
        
        # 如果没有扩展名的文件也检查
        if not audio_exists:
            audio_path = audio_dir / file_id
            if audio_path.exists():
                audio_exists = True
                entry['audio_file'] = file_id
        
        # 统计和过滤
        if video_exists and audio_exists:
            # 两个文件都存在，添加到有效条目
            valid_entries.append(entry)
        elif video_exists and not audio_exists:
            missing_audio_count += 1
            logger.debug(f"缺失音频: {file_id}")
        elif not video_exists and audio_exists:
            missing_video_count += 1
            logger.debug(f"缺失视频: {file_id}")
        else:
            missing_both_count += 1
            logger.debug(f"音视频都缺失: {file_id}")
    
    # 保存过滤后的JSON
    logger.info(f"正在保存过滤后的数据到: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    logger.info("="*50)
    logger.info("数据过滤统计报告")
    logger.info("="*50)
    logger.info(f"原始条目数: {len(entries)}")
    logger.info(f"有效条目数: {len(valid_entries)}")
    logger.info(f"仅缺失视频: {missing_video_count}")
    logger.info(f"仅缺失音频: {missing_audio_count}")
    logger.info(f"音视频都缺失: {missing_both_count}")
    logger.info(f"有效率: {len(valid_entries)/len(entries)*100:.2f}%")
    logger.info("="*50)
    
    # 保存详细报告
    report = {
        "原始条目数": len(entries),
        "有效条目数": len(valid_entries),
        "仅缺失视频": missing_video_count,
        "仅缺失音频": missing_audio_count,
        "音视频都缺失": missing_both_count,
        "有效率": f"{len(valid_entries)/len(entries)*100:.2f}%"
    }
    
    report_path = Path(output_json_path).parent / "filter_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return valid_entries


def main():
    """主函数"""
    
    # ========== 配置参数 ==========
    # 根据你的实际路径修改这些参数
    
    # 训练集
    train_json = "merg_data/train.json"
    train_video_dir = "merg_data/train/video"
    train_audio_dir = "merg_data/train/audio"
    train_output_json = "merg_data/train_filtered.json"
    
    # 测试集
    test_json = "merg_data/test.json"
    test_video_dir = "merg_data/test_video"
    test_audio_dir = "merg_data/test_audio"
    test_output_json = "merg_data/test_filtered.json"
    
    # ========== 处理训练集 ==========
    if Path(train_json).exists():
        logger.info("开始处理训练集...")
        filter_missing_files(
            train_json,
            train_video_dir,
            train_audio_dir,
            train_output_json
        )
        logger.info(f"训练集处理完成！过滤后的文件保存在: {train_output_json}")
    else:
        logger.warning(f"训练集JSON文件不存在: {train_json}")
    
    # ========== 处理测试集 ==========
    if Path(test_json).exists():
        logger.info("\n开始处理测试集...")
        filter_missing_files(
            test_json,
            test_video_dir,
            test_audio_dir,
            test_output_json
        )
        logger.info(f"测试集处理完成！过滤后的文件保存在: {test_output_json}")
    else:
        logger.warning(f"测试集JSON文件不存在: {test_json}")
    
    logger.info("\n所有数据处理完成！")
    logger.info("现在可以使用 train_filtered.json 和 test_filtered.json 进行训练了")


if __name__ == "__main__":
    main()
