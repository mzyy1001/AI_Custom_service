import os
import re
import pandas as pd

# ====== 配置区 ======
excel_path   = "raw_data.xlsx"   # Excel 文件路径
outdir       = "./csv_out"            # 输出文件夹
skip_sheets  = 2                      # 跳过前两个工作表
encoding     = "utf-8-sig"            # CSV 编码（Excel 友好）
keep_index   = False                  # 导出时是否保留索引
prefix_order = True                   # 文件名前是否加序号以保留表顺序
drop_all_na_rows = False              # 是否丢弃整行为空的行
# ===================

def sanitize_filename(name: str) -> str:
    """将字符串转成安全的文件名。"""
    if name is None:
        name = "sheet"
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)  # 非法字符
    name = re.sub(r"\s+", "_", name)            # 连续空白
    return name[:120] if name else "sheet"

# 读取工作簿并取得表名顺序
xls = pd.ExcelFile(excel_path, engine="calamine")
sheet_names = xls.sheet_names



# 需要导出的表（跳过前两个）
export_names = sheet_names[skip_sheets:] if len(sheet_names) > skip_sheets else []

os.makedirs(outdir, exist_ok=True)

exported = 0
for ordinal, sheet in enumerate(export_names, start=skip_sheets):
    # 读取表
    df  = pd.read_excel(excel_path, sheet_name=sheet, engine="calamine")
    if df.empty:
        print(f"[SKIP] {sheet} is empty, skipping.")
        continue

    # 可选：丢弃整行为空
    if drop_all_na_rows:
        df = df.dropna(how="all")

    # 空表跳过
    if (df.dropna(how="all").empty and df.empty):
        continue

    safe_name = sanitize_filename(sheet)
    prefix = f"{ordinal:02d}_" if prefix_order else ""
    out_path = os.path.join(outdir, f"{prefix}{safe_name}.csv")

    # 导出 CSV
    df.to_csv(out_path, index=keep_index, encoding=encoding)
    exported += 1
    print(f"[OK] {sheet} -> {out_path} ({len(df)} rows, {len(df.columns)} cols)")

print(f"[DONE] Exported {exported} sheet(s) to: {os.path.abspath(outdir)}")
