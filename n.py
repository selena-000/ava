"""
AvaMERG 数据集预处理脚本 - 调试版本
功能：过滤掉所有缺失音频或视频的条目，生成干净的JSON文件
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import logging
import random

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_filter_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_directory_contents(directory_path, file_type=""):
    """检查目录内容"""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        logger.error(f"{file_type}目录不存在: {directory_path}")
        return False
    
    files = list(dir_path.iterdir())
    logger.info(f"{file_type}目录 {directory_path} 包含 {len(files)} 个文件")
    
    # 显示前10个文件作为示例
    if files:
        logger.info(f"前10个{file_type}文件示例:")
        for f in files[:10]:
            logger.info(f"  - {f.name}")
    else:
        logger.warning(f"{file_type}目录为空！")
    
    return True


def filter_missing_files(
    json_path,
    video_dir, 
    audio_dir,
    output_json_path
):
    """
    过滤掉缺失文件的条目
    """
    
    # 首先检查目录
    logger.info("="*50)
    logger.info("检查目录结构")
    logger.info("="*50)
    
    video_dir_exists = check_directory_contents(video_dir, "视频")
    audio_dir_exists = check_directory_contents(audio_dir, "音频")
    
    if not video_dir_exists or not audio_dir_exists:
        logger.error("目录检查失败，请确认路径配置正确！")
        return None
    
    # 读取原始JSON
    logger.info(f"\n正在读取JSON文件: {json_path}")
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
    
    # 显示JSON结构示例
    if entries:
        logger.info("\nJSON条目结构示例:")
        sample_entry = entries[0]
        logger.info(f"第一个条目的键: {list(sample_entry.keys())}")
        logger.info(f"示例条目: {json.dumps(sample_entry, indent=2, ensure_ascii=False)[:500]}...")
    
    # 转换为Path对象
    video_dir = Path(video_dir)
    audio_dir = Path(audio_dir)
    
    # 获取目录中的所有文件（不含扩展名）
    video_files = {f.stem: f.name for f in video_dir.iterdir() if f.is_file()}
    audio_files = {f.stem: f.name for f in audio_dir.iterdir() if f.is_file()}
    
    logger.info(f"\n实际视频文件数: {len(video_files)}")
    logger.info(f"实际音频文件数: {len(audio_files)}")
    
    # 过滤有效条目
    valid_entries = []
    missing_video_count = 0
    missing_audio_count = 0
    missing_both_count = 0
    
    # 用于详细调试的样本
    missing_video_samples = []
    missing_audio_samples = []
    found_samples = []
    
    for i, entry in enumerate(tqdm(entries, desc="检查文件完整性")):
        # 获取文件ID
        file_id = None
        id_field = None
        for key in ['id', 'video_id', 'audio_id', 'file_id', 'name', 'filename']:
            if key in entry:
                file_id = str(entry[key])
                id_field = key
                break
        
        if not file_id:
            logger.warning(f"条目 {i}: 无法找到ID字段: {list(entry.keys())}")
            continue
        
        # 去除可能的文件扩展名
        file_id_stem = Path(file_id).stem
        
        # 检查视频文件是否存在
        video_exists = file_id_stem in video_files
        if video_exists:
            entry['video_file'] = video_files[file_id_stem]
        
        # 检查音频文件是否存在
        audio_exists = file_id_stem in audio_files
        if audio_exists:
            entry['audio_file'] = audio_files[file_id_stem]
        
        # 统计和过滤
        if video_exists and audio_exists:
            valid_entries.append(entry)
            if len(found_samples) < 3:
                found_samples.append(f"{file_id} (使用字段: {id_field})")
        elif video_exists and not audio_exists:
            missing_audio_count += 1
            if len(missing_audio_samples) < 5:
                missing_audio_samples.append(file_id)
        elif not video_exists and audio_exists:
            missing_video_count += 1
            if len(missing_video_samples) < 5:
                missing_video_samples.append(file_id)
        else:
            missing_both_count += 1
            if i < 5:  # 只记录前5个
                logger.debug(f"音视频都缺失: {file_id} (字段: {id_field})")
    
    # 显示调试信息
    logger.info("\n="*50)
    logger.info("调试信息")
    logger.info("="*50)
    
    if found_samples:
        logger.info("成功匹配的文件示例:")
        for sample in found_samples:
            logger.info(f"  ✓ {sample}")
    
    if missing_video_samples:
        logger.info("\n缺失视频的ID示例:")
        for sample in missing_video_samples:
            logger.info(f"  ✗ {sample}")
            # 检查是否有相似的文件名
            similar = [v for v in video_files.keys() if sample.lower() in v.lower() or v.lower() in sample.lower()]
            if similar:
                logger.info(f"    可能的匹配: {similar[:3]}")
    
    if missing_audio_samples:
        logger.info("\n缺失音频的ID示例:")
        for sample in missing_audio_samples:
            logger.info(f"  ✗ {sample}")
            # 检查是否有相似的文件名
            similar = [a for a in audio_files.keys() if sample.lower() in a.lower() or a.lower() in sample.lower()]
            if similar:
                logger.info(f"    可能的匹配: {similar[:3]}")
    
    # 随机检查一些ID的匹配情况
    if entries and len(entries) > 10:
        logger.info("\n随机抽样检查:")
        random_indices = random.sample(range(len(entries)), min(5, len(entries)))
        for idx in random_indices:
            entry = entries[idx]
            for key in ['id', 'video_id', 'audio_id', 'file_id', 'name', 'filename']:
                if key in entry:
                    file_id = str(entry[key])
                    file_id_stem = Path(file_id).stem
                    logger.info(f"  条目{idx}: ID={file_id}, Stem={file_id_stem}")
                    logger.info(f"    视频存在: {file_id_stem in video_files}")
                    logger.info(f"    音频存在: {file_id_stem in audio_files}")
                    break
    
    # 保存过滤后的JSON
    logger.info(f"\n正在保存过滤后的数据到: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    logger.info("\n="*50)
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
        "有效率": f"{len(valid_entries)/len(entries)*100:.2f}%",
        "视频目录文件数": len(video_files),
        "音频目录文件数": len(audio_files),
        "缺失视频样本": missing_video_samples[:10],
        "缺失音频样本": missing_audio_samples[:10]
    }
    
    report_path = Path(output_json_path).parent / "filter_report_debug.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return valid_entries


def main():
    """主函数"""
    
    # ========== 配置参数 ==========
    # 请确认这些路径是否正确！
    
    # 训练集
    train_json = "merg_data/train.json"
    train_video_dir = "merg_data/train/video"  # 确认这个路径
    train_audio_dir = "merg_data/train/audio"  # 确认这个路径
    train_output_json = "merg_data/train_filtered.json"
    
    # 测试集
    test_json = "merg_data/test.json"
    test_video_dir = "merg_data/test_video"  # 确认这个路径
    test_audio_dir = "merg_data/test_audio"  # 确认这个路径
    test_output_json = "merg_data/test_filtered.json"
    
    logger.info("="*50)
    logger.info("开始数据过滤处理")
    logger.info("="*50)
    logger.info(f"当前工作目录: {os.getcwd()}")
    
    # ========== 处理训练集 ==========
    if Path(train_json).exists():
        logger.info("\n开始处理训练集...")
        filter_missing_files(
            train_json,
            train_video_dir,
            train_audio_dir,
            train_output_json
        )
        logger.info(f"训练集处理完成！")
    else:
        logger.error(f"训练集JSON文件不存在: {train_json}")
    
    # ========== 处理测试集 ==========
    if Path(test_json).exists():
        logger.info("\n开始处理测试集...")
        filter_missing_files(
            test_json,
            test_video_dir,
            test_audio_dir,
            test_output_json
        )
        logger.info(f"测试集处理完成！")
    else:
        logger.error(f"测试集JSON文件不存在: {test_json}")
    
    logger.info("\n所有数据处理完成！")
    logger.info("请查看 filter_report_debug.json 了解详细信息")


if __name__ == "__main__":
    main()
