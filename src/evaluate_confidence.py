import json
import statistics
import sys
from pathlib import Path

from confidence import classify_with_margin
from load_data import load_emails

DEFAULT_FILES = ["data/emails.jsonl", "data/emails_test.jsonl"]
RESULTS_PATH = Path("eval/results_margin.json")
THRESHOLDS = [3, 6, 9, 12, 15, 18, 21]


def run_all(emails):
    rows = []
    for i, email in enumerate(emails, start=1):
        try:
            result, margin, rival = classify_with_margin(email["subject"], email["body"])
            row = {
                "id": email["id"],
                "subject": email["subject"],
                "true": email["label"],
                "pred": result.label,
                "said_confidence": result.confidence,
                "margin": margin,
                "rival": rival,
            }
        except Exception as e:
            row = {
                "id": email["id"],
                "subject": email["subject"],
                "true": email["label"],
                "pred": "error",
                "said_confidence": 0.0,
                "margin": 0.0,
                "rival": None,
                "error": str(e),
            }
        rows.append(row)
        print(f"{i}/{len(emails)} işlendi", end="\r")
    print()
    return rows


def print_report(rows):
    wrong = [r for r in rows if r["true"] != r["pred"]]
    correct = [r for r in rows if r["true"] == r["pred"]]

    print(f"\nToplam: {len(rows)} | Doğru: {len(correct)} | Yanlış: {len(wrong)}")

    margins_correct = [r["margin"] for r in correct]
    print("\nMarj dağılımı (doğru bilinenler):")
    print(f"  en küçük: {min(margins_correct):.1f} | ortanca: {statistics.median(margins_correct):.1f} | en büyük: {max(margins_correct):.1f}")

    print("\nYanlış bilinenlerin marjı:")
    for r in sorted(wrong, key=lambda r: r["margin"]):
        print(f"  marj {r['margin']:5.1f} | gerçek: {r['true']:8} tahmin: {r['pred']:8} rakip: {r['rival']} | {r['subject']}")

    print("\nEn düşük marjlı 12 e-posta (en kararsız olanlar):")
    for r in sorted(rows, key=lambda r: r["margin"])[:12]:
        durum = "DOĞRU " if r["true"] == r["pred"] else "YANLIŞ"
        print(f"  marj {r['margin']:5.1f} | {durum} | {r['pred']:8} rakip: {r['rival']} | {r['subject']}")

    print("\nEşik tablosu: marj eşiğin altındaysa insana gönder")
    print(f"{'eşik':>6} {'insana giden':>14} {'oran':>7} {'yakalanan hata':>16}")
    for t in THRESHOLDS:
        flagged = [r for r in rows if r["margin"] < t]
        caught = [r for r in flagged if r["true"] != r["pred"]]
        print(f"{t:>6} {len(flagged):>14} {len(flagged) / len(rows):>7.0%} {len(caught):>8} / {len(wrong)}")


if __name__ == "__main__":
    files = sys.argv[1:] or DEFAULT_FILES
    emails = []
    for f in files:
        emails += load_emails(Path(f))
    print(f"Veri: {files} | E-posta sayısı: {len(emails)}")

    rows = run_all(emails)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sonuçlar kaydedildi: {RESULTS_PATH}")

    print_report(rows)
