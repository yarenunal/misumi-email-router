import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/emails.jsonl")


def load_emails(path=DATA_PATH):
    """Dosyayı oku, her satırı bir sözlüğe çevir, liste olarak döndür."""
    emails = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            email = json.loads(line)
            emails.append(email)
    return emails


if __name__ == "__main__":
    emails = load_emails()
    print("Toplam e-posta:", len(emails))

    labels = [e["label"] for e in emails]
    print("Sınıf dağılımı:", Counter(labels))

    hard_count = sum(1 for e in emails if e["hard"])
    print("Zor örnek sayısı:", hard_count)

    first = emails[0]
    print("İlk e-posta konusu:", first["subject"])
    print("İlk e-posta etiketi:", first["label"])
