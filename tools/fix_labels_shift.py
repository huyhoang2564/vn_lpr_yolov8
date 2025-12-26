from pathlib import Path
import shutil

LABEL_DIR = Path(r"E:\VNUK_Project\vn_lpr_yolov8\dataset\char\labels")
SHIFT = +1          # nếu bị lùi 1 thì +1; nếu bị tăng 1 thì -1
NC = 31             # số class trong YAML

BACKUP_DIR = LABEL_DIR.parent / "_backup_labels"

def fix_one_file(p: Path):
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            out.append(line)
            continue
        cid = int(float(parts[0]))
        new_id = cid + SHIFT
        if new_id < 0 or new_id >= NC:
            raise ValueError(f"Class id out of range in {p}: {cid} -> {new_id}")
        parts[0] = str(new_id)
        out.append(" ".join(parts))
    p.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

def main():
    if not LABEL_DIR.exists():
        raise FileNotFoundError(LABEL_DIR)

    # backup
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(LABEL_DIR, BACKUP_DIR)
    print("✅ Backup labels to:", BACKUP_DIR)

    # fix all .txt
    files = list(LABEL_DIR.rglob("*.txt"))
    print("Found label files:", len(files))

    for f in files:
        fix_one_file(f)

    print(f"✅ Done! Applied SHIFT={SHIFT} to all labels.")
    print("Now retrain char model from scratch (recommended).")

if __name__ == "__main__":
    main()
