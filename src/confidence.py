import json

from classify import Classification
from probe_logprobs import ask_with_logprobs

LABELS = ["order", "quote", "return", "support", "other"]


def label_margin(logprobs):
    """Etiket token'ını bul, seçilen etiket ile en yakın rakip arasındaki farkı hesapla.

    Döndürür: (marj, en_yakın_rakip_etiket)
    Marj ne kadar küçükse model o kadar kararsızdır.
    """
    start = next(i for i, item in enumerate(logprobs) if item["token"] == "label")

    label_item = None
    for item in logprobs[start + 1 :]:
        if item["token"] in LABELS:
            label_item = item
            break
    if label_item is None:
        return None, None

    chosen = label_item["token"]
    chosen_logprob = label_item["logprob"]

    # Alternatifler arasından sadece geçerli etiketleri al (seçilen hariç)
    rivals = {
        alt["token"]: alt["logprob"]
        for alt in label_item["top_logprobs"]
        if alt["token"] in LABELS and alt["token"] != chosen
    }

    if rivals:
        rival = max(rivals, key=rivals.get)
        return chosen_logprob - rivals[rival], rival

    # Hiçbir rakip ilk 5 alternatifte yoksa marjın alt sınırını kullan
    lowest = min(alt["logprob"] for alt in label_item["top_logprobs"])
    return chosen_logprob - lowest, None


def classify_with_margin(subject, body):
    """E-postayı sınıflandır ve marj bilgisini de döndür."""
    data = ask_with_logprobs(subject, body)
    result = Classification.model_validate_json(data["message"]["content"])
    margin, rival = label_margin(data["logprobs"])
    return result, margin, rival
