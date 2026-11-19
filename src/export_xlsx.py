"""
One xlsx with all samples, in the layout of the thesis table.

Usage:
    python src/export_xlsx.py --samples 1 2 3 4 5 6 8 10 11 12 13 14
"""
import argparse
import csv
from collections import defaultdict

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, Side

from detect_attachments import DATA_DIR, PROCESSED_DIR

# Read from the picture, so CHECK spelling. Add groups 7 and 9 if you have them.
TITLES = {
    "1": "بیوتیفیل شوفو / بیضی",
    "2": "پکبل سارمکو / مستطیل افقی با بول",
    "3": "بیوتیفیل شوفو / مستطیل افقی",
    "4": "فلوینل سارمکو / مستطیل عمودی",
    "5": "فلوینل سارمکو / بیضی",
    "6": "پکبل سارمکو / مستطیل عمودی",
    "8": "بیوتیفیل شوفو / مستطیل افقی با بول",
    "10": "فلوینل سارمکو / مستطیل افقی",
    "11": "بیوتیفیل شوفو / مستطیل عمودی",
    "12": "فلوینل سارمکو / مستطیل افقی با بول",
    "13": "پکبل سارمکو / بیضی",
    "14": "پکبل سارمکو / مستطیل افقی",
}
HEADERS = ["اتچمنت", "defect", "excess", "RMS", "pre volume", "post volume",
           "wear loss", "wear percent", "SBS (MPa)", "ARI", "failure mode"]
N_TEETH = 14
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
FAIL_MAP = {"a": "ادهزیو", "adhesive": "ادهزیو",
            "c": "کوهزیو", "cohesive": "کوهزیو",
            "m": "مختلط", "mixed": "مختلط"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_manual():
    """Optional. Returns {(sample, tooth): {...}} or {} if no manual file exists."""
    xlsx, csvp = DATA_DIR / "manual_all.xlsx", DATA_DIR / "manual_all.csv"
    if xlsx.exists():
        ws = load_workbook(xlsx, data_only=True).active
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    elif csvp.exists():
        raw = csvp.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1256")
        lines = text.splitlines()
        delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
        rows = list(csv.reader(lines, delimiter=delim))
    else:
        return {}

    head = [str(h).strip().lower() for h in rows[0]]
    col = {name: i for i, name in enumerate(head)}
    key = "sample" if "sample" in col else head[0]

    def get(r, name):
        i = col.get(name)
        return r[i] if i is not None and i < len(r) else None

    out, count = {}, defaultdict(int)
    for r in rows[1:]:
        label = get(r, key)
        if label is None or str(label).strip() == "":
            continue
        s = str(label).strip().split(".")[0]
        count[s] += 1
        t = count[s]
        fm = str(get(r, "failure_mode") or "").strip()
        fm = FAIL_MAP.get(fm.lower(), fm)
        if "?" in fm:
            print(f"[WARNING] sample {s} tooth {t}: failure mode '{fm}' is broken. "
                  f"Retype it or use a / c / m.")
        out[(s, t)] = dict(defect=get(r, "defect"), excess=get(r, "excess"),
                           rms=get(r, "rms"), sbs=get(r, "sbs"),
                           ari=get(r, "ari"), failure_mode=fm or None)
    for s, c in count.items():
        if c != N_TEETH:
            print(f"[WARNING] manual table: sample {s} has {c} rows, expected {N_TEETH}")
    return out


def main(args):
    manual = load_manual()
    if not manual:
        print("[INFO] no manual_all file: defect/excess/RMS/SBS/ARI/failure mode stay empty")

    wb = Workbook()
    ws = wb.active
    ws.title = "All samples"
    thin = Side(style="thin", color="BBBBBB")
    r = 1
    for s in args.samples:
        vols = {int(v["tooth"]): v
                for v in read_csv(PROCESSED_DIR / s / "volumes_matched.csv")}
        if not vols:
            print(f"[WARNING] sample {s}: no volumes_matched.csv -> volume columns empty")

        ws.cell(r, 1, f"گروه {s.translate(FA)}: {TITLES.get(s, '')}").font = Font(bold=True, size=12)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(HEADERS))
        ws.cell(r, 1).alignment = Alignment(horizontal="left")
        r += 1
        for c, h in enumerate(HEADERS, start=1):
            ws.cell(r, c, h).font = Font(bold=True)
        hdr = r
        r += 1

        first = r
        for t in range(1, N_TEETH + 1):
            v, m = vols.get(t, {}), manual.get((s, t), {})
            ws.cell(r, 1, f"{s}.{t}").number_format = "@"
            ws.cell(r, 2, num(m.get("defect")))
            ws.cell(r, 3, num(m.get("excess")))
            ws.cell(r, 4, num(m.get("rms")))
            ws.cell(r, 5, num(v.get("pre_volume_mm3")))
            ws.cell(r, 6, num(v.get("post_volume_mm3")))
            ws.cell(r, 7, f'=IF(AND(ISNUMBER(E{r}),ISNUMBER(F{r})),E{r}-F{r},"")')
            ws.cell(r, 8, f'=IF(AND(ISNUMBER(G{r}),E{r}>0),G{r}/E{r},"")')
            ws.cell(r, 9, num(m.get("sbs")))
            ws.cell(r, 10, num(m.get("ari")))
            ws.cell(r, 11, m.get("failure_mode") or None)
            for c in (5, 6, 7):
                ws.cell(r, c).number_format = "0.0000"
            ws.cell(r, 8).number_format = "0.00%"
            r += 1
        last = r - 1

        ws.cell(r, 1, "میانگین").font = Font(bold=True)
        for c in range(2, 10):
            L = ws.cell(hdr, c).column_letter
            cell = ws.cell(r, c, f'=IFERROR(AVERAGE({L}{first}:{L}{last}),"")')
            cell.font = Font(bold=True)
            cell.number_format = "0.00%" if c == 8 else "0.00"

        for row in ws.iter_rows(min_row=hdr, max_row=r, max_col=len(HEADERS)):
            for cell in row:
                cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
                cell.alignment = Alignment(horizontal="center")
        r += 2

    for col in "ABCDEFGHIJK":
        ws.column_dimensions[col].width = 14
    dst = PROCESSED_DIR / "results_all_samples.xlsx"
    wb.save(dst)
    print(f"Saved {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    main(ap.parse_args())