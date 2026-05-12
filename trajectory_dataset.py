import random
import pickle
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Loads offline STARTrack trajectory feature files and samples fixed-length snippets.

    Each .pt file is expected to contain a dict with a "frames" list. Each frame
    must provide:
      topk_feats: [K, C]
      topk_scores: [K]
      topk_ious: [K]            # required when target_source="iou"
      oracle_best_idx: int
      gt_feat: [C]
      gt_inside_crop: bool
    """

    def __init__(
        self,
        feature_dir,
        snippet_len=32,
        hard_prob=0.7,
        samples_per_epoch=None,
        recursive=True,
        seed=0,
        target_source="iou",
        iou_thresh=0.3,
        hard_list=None,
        full_feature_dirs=None,
        hard_feature_root=None,
        hard_pool_ratio=0.7,
        full_pool_ratio=0.3,
        full_hard_prob=0.2,
        exclude_hard_from_full=True,
        disable_cache=False,
    ):
        self.feature_dir = Path(feature_dir)
        self.snippet_len = int(snippet_len)
        self.hard_prob = float(hard_prob)
        self.full_hard_prob = float(full_hard_prob)
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(seed)
        self.target_source = str(target_source)
        self.iou_thresh = float(iou_thresh)
        self.hard_pool_ratio = float(hard_pool_ratio)
        self.full_pool_ratio = float(full_pool_ratio)
        self.exclude_hard_from_full = bool(exclude_hard_from_full)
        self.disable_cache = bool(disable_cache)
        self.mixed_pool = hard_list is not None or bool(full_feature_dirs)
        self.hard_index = []
        self.full_index = []
        self.bad_paths = set()

        if self.target_source not in ("iou", "oracle"):
            raise ValueError(f"Unknown target_source: {self.target_source}")

        if self.mixed_pool:
            self._build_mixed_index(
                hard_list=hard_list,
                full_feature_dirs=full_feature_dirs,
                hard_feature_root=hard_feature_root,
                recursive=recursive,
            )
            self.index = self.hard_index + self.full_index
            if self.hard_pool_ratio > 0.0 and len(self.hard_index) == 0:
                raise RuntimeError("hard_pool_ratio > 0 but hard_index is empty.")
            if self.full_pool_ratio > 0.0 and len(self.full_index) == 0:
                raise RuntimeError("full_pool_ratio > 0 but full_index is empty.")
            if (self.hard_pool_ratio > 0.0 and len(self.hard_index) > 0) or (
                self.full_pool_ratio > 0.0 and len(self.full_index) > 0
            ):
                pass
            else:
                raise RuntimeError("Both hard/full sampling pools are disabled or empty.")
            if self.samples_per_epoch is None:
                self.samples_per_epoch = max(
                    len(self.index),
                    sum(max(1, item["num_frames"] // self.snippet_len) for item in self.index),
                )
            return

        pattern = "**/*.pt" if recursive else "*.pt"
        self.files = sorted(self.feature_dir.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No .pt trajectory files found under {self.feature_dir}")

        if self.target_source == "iou":
            cache_file = self.feature_dir / f"dataset_index_cache_iou_thr{self.iou_thresh:.2f}.pkl"
        else:
            cache_file = self.feature_dir / "dataset_index_cache_oracle.pkl"
        
        if (not self.disable_cache) and cache_file.exists():
            print(f"📦 检测到数据集缓存，正在极速加载: {cache_file}")
            with open(cache_file, "rb") as f:
                self.index = pickle.load(f)
            print(f"✅ 成功加载 {len(self.index)} 个视频的索引！")
        else:
            print(f"⏳ 首次读取特征库，正在扫描 {len(self.files)} 个文件构建索引 (耗时较长，请耐心等待)...")
            self.index = []
            skipped_bad_files = 0
            
            # 加上 tqdm 进度条，这样你就知道它到底卡在哪了
            for path in tqdm(self.files, desc="Building Dataset Index", dynamic_ncols=True):
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as e:
                    # 损坏文件我们在之前的脚本里应该已经删了，如果还有就跳过
                    continue 
                
                frames = data.get("frames", None) if isinstance(data, dict) else None
                if not frames:
                    continue
                
                hard_frames = []
                try:
                    for i, frame in enumerate(frames):
                        if self._is_hard_frame(frame):
                            hard_frames.append(i)
                except Exception as e:
                    skipped_bad_files += 1
                    print(f"[WARN] Skipping bad trajectory file during index build: {path} ({e})")
                    continue

                self.index.append({
                    "path": path,
                    "num_frames": len(frames),
                    "hard_frames": hard_frames,
                    "pool": "single",
                })
                
            # 保存缓存，下次秒进！
            if not self.disable_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.index, f)
                print(f"💾 索引构建完成，已保存缓存至: {cache_file}")
            if skipped_bad_files > 0:
                print(f"[WARN] Skipped {skipped_bad_files} bad trajectory files while building dataset index.")
        if not self.index:
            raise RuntimeError(f"No valid trajectory files with non-empty frames found under {self.feature_dir}")
        if self.samples_per_epoch is None:
            self.samples_per_epoch = max(len(self.index), sum(max(1, item["num_frames"] // self.snippet_len) for item in self.index))

    def __len__(self):
        return int(self.samples_per_epoch)

    def _build_mixed_index(self, hard_list, full_feature_dirs, hard_feature_root, recursive=True):
        print("[INFO] Building mixed-pool trajectory dataset index (cache disabled for mixed mode).")
        hard_paths = self._read_hard_list(hard_list, hard_feature_root) if hard_list is not None else []
        hard_path_set = {self._path_key(path) for path in hard_paths}

        self.hard_index = self._build_index_from_paths(hard_paths, pool="hard", desc="Building Hard Pool")

        full_paths = []
        pattern = "**/*.pt" if recursive else "*.pt"
        for feature_dir in full_feature_dirs or []:
            feature_dir = Path(feature_dir)
            if not feature_dir.exists():
                print(f"[WARN] full_feature_dir does not exist, skip: {feature_dir}")
                continue
            full_paths.extend(sorted(feature_dir.glob(pattern)))

        if self.exclude_hard_from_full:
            before = len(full_paths)
            full_paths = [path for path in full_paths if self._path_key(path) not in hard_path_set]
            print(f"[INFO] exclude_hard_from_full removed {before - len(full_paths)} files from full pool.")

        self.full_index = self._build_index_from_paths(full_paths, pool="full", desc="Building Full Pool")
        print(f"[INFO] hard_pool_size={len(self.hard_index)}, full_pool_size={len(self.full_index)}")

    def _read_hard_list(self, hard_list, hard_feature_root):
        hard_list = Path(hard_list)
        if not hard_list.exists():
            raise FileNotFoundError(f"hard_list not found: {hard_list}")
        root = Path(hard_feature_root) if hard_feature_root is not None else None
        paths = []
        seen = set()
        with open(hard_list, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line)
                if not path.is_absolute() and root is not None:
                    path = root / path
                if not path.exists():
                    print(f"[WARN] hard_list entry does not exist, skip: {path}")
                    continue
                key = self._path_key(path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
        return paths

    def _path_key(self, path):
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(Path(path).absolute())

    def _build_index_from_paths(self, paths, pool, desc):
        index = []
        skipped_bad_files = 0
        for path in tqdm(list(paths), desc=desc, dynamic_ncols=True):
            meta = self._build_meta(path, pool=pool)
            if meta is None:
                skipped_bad_files += 1
                continue
            index.append(meta)
        if skipped_bad_files > 0:
            print(f"[WARN] Skipped {skipped_bad_files} bad {pool} trajectory files while building index.")
        return index

    def _build_meta(self, path, pool):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[WARN] Skipping bad trajectory file: {path} ({e})")
            return None

        frames = data.get("frames", None) if isinstance(data, dict) else None
        if not frames:
            print(f"[WARN] Skipping trajectory file with empty frames: {path}")
            return None

        hard_frames = []
        try:
            for i, frame in enumerate(frames):
                if self._is_hard_frame(frame):
                    hard_frames.append(i)
        except Exception as e:
            print(f"[WARN] Skipping bad trajectory file during index build: {path} ({e})")
            return None

        return {
            "path": Path(path),
            "num_frames": len(frames),
            "hard_frames": hard_frames,
            "pool": pool,
        }

    def _sample_start(self, meta, hard_prob=None):
        if hard_prob is None:
            hard_prob = self.hard_prob
        n = int(meta["num_frames"])
        max_start = max(n - self.snippet_len, 0)

        use_hard = self.rng.random() < float(hard_prob) and len(meta["hard_frames"]) > 0
        if use_hard:
            hard_idx = self.rng.choice(meta["hard_frames"])
            lo = max(0, hard_idx - self.snippet_len + 1)
            hi = min(hard_idx, max_start)
            return self.rng.randint(lo, hi) if hi >= lo else min(max(hard_idx, 0), max_start)

        return self.rng.randint(0, max_start) if max_start > 0 else 0

    def _is_hard_frame(self, frame):
        gt_inside_crop = bool(frame.get("gt_inside_crop", False))
        if not gt_inside_crop:
            return False

        if self.target_source == "oracle":
            return int(frame.get("oracle_best_idx", 0)) != 0

        if "topk_ious" not in frame:
            raise KeyError(
                "Frame is missing required field 'topk_ious' while target_source='iou'. "
                "Use --target-source oracle for old oracle_best_idx data."
            )

        topk_ious = torch.as_tensor(frame["topk_ious"], dtype=torch.float32)
        if topk_ious.dim() != 1:
            raise ValueError(f"topk_ious must be [K], got {tuple(topk_ious.shape)}")

        best_iou, best_idx_by_iou = torch.max(topk_ious, dim=-1)
        if float(best_iou.item()) < self.iou_thresh:
            return False

        has_apply_label = "apply_label" in frame
        has_recoverable_label = "recoverable_label" in frame
        if has_apply_label or has_recoverable_label:
            return (
                int(frame.get("apply_label", 0)) == 1
                or int(frame.get("recoverable_label", 0)) == 1
                or int(best_idx_by_iou.item()) != 0
            )

        return int(best_idx_by_iou.item()) != 0

    def _frame_to_tensor(self, frame):
        topk_feats = torch.as_tensor(frame["topk_feats"], dtype=torch.float32)
        topk_scores = torch.as_tensor(frame["topk_scores"], dtype=torch.float32)
        gt_feat = torch.as_tensor(frame["gt_feat"], dtype=torch.float32)
        gt_inside_crop = torch.as_tensor(bool(frame["gt_inside_crop"]), dtype=torch.bool)

        if self.target_source == "oracle":
            if "oracle_best_idx" not in frame:
                raise KeyError("Frame is missing required field 'oracle_best_idx' while target_source='oracle'.")
            oracle_best_idx = torch.as_tensor(int(frame["oracle_best_idx"]), dtype=torch.long)
        else:
            oracle_best_idx = torch.as_tensor(int(frame.get("oracle_best_idx", 0)), dtype=torch.long)

        has_topk_ious = "topk_ious" in frame
        if has_topk_ious:
            topk_ious = torch.as_tensor(frame["topk_ious"], dtype=torch.float32)
        elif self.target_source == "iou":
            raise KeyError(
                "Frame is missing required field 'topk_ious' while target_source='iou'. "
                "Use --target-source oracle for old oracle_best_idx data."
            )
        else:
            topk_ious = torch.zeros(topk_feats.size(0), dtype=torch.float32)

        if "oracle_labels" in frame:
            oracle_labels = torch.as_tensor(frame["oracle_labels"], dtype=torch.float32)
        else:
            oracle_labels = torch.zeros(topk_feats.size(0), dtype=torch.float32)

        if topk_feats.dim() != 2:
            raise ValueError(f"topk_feats must be [K, C], got {tuple(topk_feats.shape)}")
        if topk_scores.dim() != 1 or topk_scores.size(0) != topk_feats.size(0):
            raise ValueError("topk_scores must be [K] and align with topk_feats")
        if topk_ious.dim() != 1 or topk_ious.size(0) != topk_feats.size(0):
            raise ValueError("topk_ious must be [K] and align with topk_feats")
        if oracle_labels.dim() != 1 or oracle_labels.size(0) != topk_feats.size(0):
            raise ValueError("oracle_labels must be [K] and align with topk_feats")
        if gt_feat.dim() != 1 or gt_feat.size(0) != topk_feats.size(1):
            raise ValueError("gt_feat must be [C] and align with topk_feats")

        return {
            "topk_feats": topk_feats,
            "topk_scores": topk_scores,
            "topk_ious": topk_ious,
            "oracle_best_idx": oracle_best_idx,
            "oracle_labels": oracle_labels,
            "gt_feat": gt_feat,
            "gt_inside_crop": gt_inside_crop,
            "has_topk_ious": torch.as_tensor(has_topk_ious, dtype=torch.bool),
        }

    def __getitem__(self, _):
        max_retry = 50
        last_error = None
        for _retry in range(max_retry):
            if self.mixed_pool:
                meta, hard_prob = self._sample_mixed_meta()
            else:
                meta = self._sample_single_meta()
                hard_prob = self.hard_prob

            try:
                data = torch.load(meta["path"], map_location="cpu", weights_only=False)
                break
            except Exception as e:
                last_error = e
                path_key = self._path_key(meta["path"])
                self.bad_paths.add(path_key)
                print(f"[WARN] skip bad pt: {meta['path']}, error={repr(e)}")
                continue
        else:
            raise RuntimeError(f"Failed to load a valid .pt after {max_retry} retries. Last error: {repr(last_error)}")

        frames = data["frames"]
        start = self._sample_start(meta, hard_prob=hard_prob)

        items = []
        last_idx = len(frames) - 1
        for offset in range(self.snippet_len):
            frame_idx = min(start + offset, last_idx)
            items.append(self._frame_to_tensor(frames[frame_idx]))

        batch = {}
        for key in items[0].keys():
            batch[key] = torch.stack([item[key] for item in items], dim=0)
        batch["pool"] = meta.get("pool", "single")
        batch["meta_path"] = str(meta["path"])
        batch["start_frame"] = torch.as_tensor(start, dtype=torch.long)
        return batch

    def _sample_single_meta(self):
        candidates = [meta for meta in self.index if self._path_key(meta["path"]) not in self.bad_paths]
        if len(candidates) == 0:
            raise RuntimeError("No valid trajectory files remain after filtering bad_paths.")
        return self.rng.choice(candidates)

    def _sample_mixed_meta(self):
        hard_candidates = [meta for meta in self.hard_index if self._path_key(meta["path"]) not in self.bad_paths]
        full_candidates = [meta for meta in self.full_index if self._path_key(meta["path"]) not in self.bad_paths]
        hard_weight = self.hard_pool_ratio if len(hard_candidates) > 0 else 0.0
        full_weight = self.full_pool_ratio if len(full_candidates) > 0 else 0.0
        total = hard_weight + full_weight
        if total <= 0.0:
            raise RuntimeError("No available sampling pool after filtering bad_paths.")
        use_hard_pool = self.rng.random() < (hard_weight / total)
        if use_hard_pool:
            return self.rng.choice(hard_candidates), self.hard_prob
        return self.rng.choice(full_candidates), self.full_hard_prob