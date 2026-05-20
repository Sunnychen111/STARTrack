#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np


def load_txt_boxes(txt_path: Path):
    """
    兼容以下格式：
    1) x y w h
    2) x,y,w,h
    3) x\t y\t w\t h
    """
    try:
        data = np.loadtxt(txt_path, delimiter=",")
    except Exception:
        data = np.loadtxt(txt_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        raise ValueError(f"{txt_path} 列数小于 4，无法作为 bbox 处理: shape={data.shape}")

    # 只保留前 4 列：x, y, w, h
    data = data[:, :4]
    return data


def convert_one_file(src_path: Path, dst_path: Path, mode: str = "astype"):
    boxes = load_txt_boxes(src_path)

    if mode == "astype":
        # 对齐 SUTrack 官方保存逻辑：np.array(data).astype(int)
        boxes_int = boxes.astype(int)
    elif mode == "round":
        boxes_int = np.round(boxes).astype(int)
    elif mode == "floor":
        boxes_int = np.floor(boxes).astype(int)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst_path, boxes_int, delimiter="\t", fmt="%d")


def main():
    parser = argparse.ArgumentParser(
        description="Convert tracking result txt bbox values to integers."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="原始 txt 结果文件夹，例如 results/tnl2k_float"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="转换后保存的文件夹，例如 results/tnl2k_int"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归处理子文件夹"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="astype",
        choices=["astype", "round", "floor"],
        help="整数转换方式。astype 对齐 SUTrack，直接截断小数；round 为四舍五入；floor 为向下取整。"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir 不存在: {input_dir}")

    txt_files = list(input_dir.rglob("*.txt")) if args.recursive else list(input_dir.glob("*.txt"))

    print(f"[INFO] Found {len(txt_files)} txt files.")

    success = 0
    failed = 0

    for txt_path in txt_files:
        rel_path = txt_path.relative_to(input_dir)
        dst_path = output_dir / rel_path

        try:
            convert_one_file(txt_path, dst_path, mode=args.mode)
            success += 1
            print(f"[OK] {rel_path}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {rel_path}: {e}")

    print("=" * 60)
    print(f"[DONE] success={success}, failed={failed}")
    print(f"[SAVE] output_dir={output_dir}")


if __name__ == "__main__":
    main()