import json
import sys
import time
from collections import Counter
from pathlib import Path

from classify import MODEL, classify_email
from load_data import load_emails

LABELS = ["order", "quote", "return", "support", "other"]
RESULTS_PATH = Path("eval/results_baseline.json")


def run_all(emails):
    """Tüm e-postaları sınıflandır, her biri için bir sonuç kaydı döndür."""
    results = []
    for i, email in enumerate(emails, start=1):
        try:
            pred = classify_email(email["subject"], email["body"])
            pred_label, confidence, reason = pred.label, pred.confidence, pred.reason
        except Exception as e:
            # Model bozuk cevap verirse program durmasın, hata olarak kaydet
            pred_label, confidence, reason = "error", 0.0, str(e)

        results.append(
            {
                "id": email["id"],
                "subject": email["subject"],
                "true": email["label"],
                "pred": pred_label,
                "confidence": confidence,
                "reason": reason,
                "hard": email["hard"],
            }
        )
        print(f"{i}/{len(emails)} işlendi", end="\r")
    print()
    return results


def accuracy(rows):
    """Doğru tahmin oranı. Liste boşsa 0 döndür."""
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if r["true"] == r["pred"])
    return correct / len(rows)


def average_confidence(rows):
    if not rows:
        return 0.0
    return sum(r["confidence"] for r in rows) / len(rows)


def print_report(results):
    print(f"\nGenel doğruluk: {accuracy(results):.0%}")

    easy = [r for r in results if not r["hard"]]
    hard = [r for r in results if r["hard"]]
    print(f"Kolay örnekler: {accuracy(easy):.0%} ({len(easy)} e-posta)")
    print(f"Zor örnekler:   {accuracy(hard):.0%} ({len(hard)} e-posta)")

    print("\nSınıf başına doğruluk:")
    for label in LABELS:
        rows = [r for r in results if r["true"] == label]
        print(f"  {label:8} {accuracy(rows):.0%}")

    print("\nKarışıklık tablosu (satır = gerçek, sütun = tahmin):")
    print(" " * 10 + "".join(f"{label:>9}" for label in LABELS))
    for true_label in LABELS:
        counts = Counter(r["pred"] for r in results if r["true"] == true_label)
        print(f"{true_label:10}" + "".join(f"{counts[p]:>9}" for p in LABELS))

    correct = [r for r in results if r["true"] == r["pred"]]
    wrong = [r for r in results if r["true"] != r["pred"]]
    print("\nGüven skoru:")
    print(f"  Doğru bildiklerinde ortalama: {average_confidence(correct):.2f}")
    print(f"  Yanlış bildiklerinde ortalama: {average_confidence(wrong):.2f}")

    print(f"\nYanlış tahminler ({len(wrong)} adet):")
    for r in wrong:
        print(f"- {r['subject']}")
        print(f"    gerçek: {r['true']} | tahmin: {r['pred']} | güven: {r['confidence']}")
        print(f"    gerekçe: {r['reason']}")


if __name__ == "__main__":
    # Kullanım: python src/evaluate.py [veri_dosyası] [sonuç_dosyası]
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/emails.jsonl")
    results_path = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_PATH

    emails = load_emails(data_path)
    print(f"Model: {MODEL} | Veri: {data_path} | E-posta sayısı: {len(emails)}")

    start = time.time()
    results = run_all(emails)
    print(f"Süre: {time.time() - start:.0f} saniye")

    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sonuçlar kaydedildi: {results_path}")

    print_report(results)
