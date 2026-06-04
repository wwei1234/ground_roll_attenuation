"""
掩码文件批量检查工具
用法：直接运行此脚本，修改下方 MASK_DIR 为你的掩码文件夹路径
"""

import os
import numpy as np

# ============================================================
# 配置：修改为你的掩码文件夹路径
MASK_DIR = r"masks"
# ============================================================

def check_mask_files(mask_dir):
    if not os.path.exists(mask_dir):
        print(f"[错误] 文件夹不存在: {mask_dir}")
        return

    npy_files = sorted([
        f for f in os.listdir(mask_dir) if f.endswith(".npy")
    ])

    if not npy_files:
        print(f"[警告] 文件夹中没有找到任何 .npy 文件: {mask_dir}")
        return

    print(f"共找到 {len(npy_files)} 个 .npy 文件，开始检查...\n")
    print("=" * 65)

    normal_list  = []
    problem_list = []

    for fname in npy_files:
        fpath = os.path.join(mask_dir, fname)

        try:
            raw = np.load(fpath, allow_pickle=True)

            # ---------- 判断各种异常情况 ----------
            issues = []

            # 1. 0维对象（可能包裹了 None 或其他对象）
            if raw.ndim == 0:
                inner = raw.item()
                if inner is None:
                    issues.append("内容为 None（0维空对象）")
                else:
                    issues.append(f"0维对象，内容类型={type(inner).__name__}")

            # 2. 对象数组（含 None 或混合类型）
            elif raw.dtype == object:
                none_count = sum(1 for x in raw.flat if x is None)
                issues.append(f"dtype=object（含 None 数量={none_count}）")

            # 3. 形状异常（空数组）
            elif raw.size == 0:
                issues.append("数组为空 size=0")

            # 4. 包含 NaN / Inf
            elif np.issubdtype(raw.dtype, np.floating):
                nan_count = int(np.isnan(raw).sum())
                inf_count = int(np.isinf(raw).sum())
                if nan_count > 0:
                    issues.append(f"包含 NaN 数量={nan_count}")
                if inf_count > 0:
                    issues.append(f"包含 Inf 数量={inf_count}")

            # ---------- 汇总结果 ----------
            if issues:
                problem_list.append((fname, raw.dtype, raw.shape, issues))
                print(f"[❌ 异常] {fname}")
                print(f"         dtype={raw.dtype}, shape={raw.shape}")
                for iss in issues:
                    print(f"         问题: {iss}")
            else:
                normal_list.append(fname)
                print(f"[✓ 正常] {fname}  dtype={raw.dtype}, shape={raw.shape}")

        except Exception as e:
            problem_list.append((fname, "N/A", "N/A", [f"加载失败: {e}"]))
            print(f"[❌ 加载失败] {fname}")
            print(f"              错误: {e}")

    # ---------- 汇总报告 ----------
    print("\n" + "=" * 65)
    print(f"检查完毕：正常 {len(normal_list)} 个，异常 {len(problem_list)} 个")

    if problem_list:
        print("\n【异常文件汇总】")
        for fname, dtype, shape, issues in problem_list:
            print(f"  · {fname}  (dtype={dtype}, shape={shape})")
            for iss in issues:
                print(f"      → {iss}")

        print("\n【建议修复方式】")
        print("  如果掩码全为0可以用下面代码重新生成（根据实际shape修改）：")
        print()
        for fname, dtype, shape, _ in problem_list:
            fpath = os.path.join(mask_dir, fname)
            if shape not in ("N/A", ()):
                print(f"  np.save(r'{fpath}', np.zeros({shape}, dtype=np.uint8))")
            else:
                print(f"  # {fname} 需要手动确认正确 shape 后再重新生成")
    else:
        print("\n所有掩码文件均正常，无需修复。")


if __name__ == "__main__":
    check_mask_files(MASK_DIR)
    input("\n按回车键退出...")