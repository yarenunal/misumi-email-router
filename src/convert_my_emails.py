import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("data/my_emails.txt")
OUTPUT_PATH = Path("data/emails_mine.jsonl")
VALID_LABELS = ["order", "quote", "return", "support", "other"]


def split_blocks(text):
    """Metni '===' satırlarından bölüp her e-postayı ayrı bir blok yap."""
    blocks = re.split(r"^===\s*$", text, flags=re.MULTILINE)
    return [b.strip() for b in blocks if b.strip()]


def parse_block(block):
    """Bir bloğu LABEL, SUBJECT, BODY alanlarına ayır."""
    fields = {"LABEL": "", "SUBJECT": "", "BODY": ""}
    current = None

    for line in block.splitlines():
        for key in fields:
            if line.startswith(key + ":"):
                current = key
                fields[key] = line[len(key) + 1 :].strip()
                break
        else:
            # Hiçbir alan adıyla başlamıyorsa, BODY'nin devamı olan satırdır
            if current == "BODY":
                fields["BODY"] += "\n" + line

    return {key: value.strip() for key, value in fields.items()}


if __name__ == "__main__":
    text = INPUT_PATH.read_text(encoding="utf-8")
    blocks = split_blocks(text)

    emails = []
    errors = []

    for number, block in enumerate(blocks, start=1):
        fields = parse_block(block)

        if fields["LABEL"] not in VALID_LABELS:
            errors.append(f"E-posta {number}: LABEL geçersiz veya boş ({fields['LABEL']!r}). Konu: {fields['SUBJECT']!r}")
            continue
        if not fields["BODY"]:
            errors.append(f"E-posta {number}: BODY boş. Konu: {fields['SUBJECT']!r}")
            continue

        emails.append(
            {
                "id": f"m{len(emails) + 1:03d}",
                "subject": fields["SUBJECT"],
                "body": fields["BODY"],
                "label": fields["LABEL"],
                "has_attachment": False,
                "hard": False,
            }
        )

    print(f"Okunan blok: {len(blocks)} | Geçerli e-posta: {len(emails)} | Hatalı: {len(errors)}")
    print("Sınıf dağılımı:", dict(Counter(e["label"] for e in emails)))

    for error in errors:
        print("HATA:", error)

    lines = [json.dumps(e, ensure_ascii=False) for e in emails]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Kaydedildi: {OUTPUT_PATH}")
