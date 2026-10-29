import json
import urllib.request
from pathlib import Path

from classify import SYSTEM_PROMPT, Classification, MODEL
from load_data import load_emails

URL = "http://localhost:11434/api/chat"


def ask_with_logprobs(subject, body):
    """Ollama sunucusuna doğrudan istek at, olasılık bilgisini de iste."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Subject: {subject}\n\nBody:\n{body}"},
        ],
        "format": Classification.model_json_schema(),
        "options": {"temperature": 0},
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


PROBE_IDS = ["e001", "e031", "t001"]


def show_label_tokens(logprobs):
    """'label' kelimesinden sonraki birkaç token'ı ve alternatiflerini yazdır."""
    start = next(i for i, item in enumerate(logprobs) if item["token"] == "label")

    for item in logprobs[start : start + 5]:
        print(f"  seçilen: {item['token']!r}  (logprob {item['logprob']:.2f})")
        for alt in item["top_logprobs"][:5]:
            print(f"      alternatif: {alt['token']!r:12} logprob {alt['logprob']:.2f}")


if __name__ == "__main__":
    all_emails = load_emails(Path("data/emails.jsonl")) + load_emails(Path("data/emails_test.jsonl"))
    by_id = {e["id"]: e for e in all_emails}

    for email_id in PROBE_IDS:
        email = by_id[email_id]
        data = ask_with_logprobs(email["subject"], email["body"])
        answer = json.loads(data["message"]["content"])

        print(f"\n=== {email_id}: {email['subject']}")
        print(f"gerçek: {email['label']} | tahmin: {answer['label']} | söylediği güven: {answer['confidence']}")
        show_label_tokens(data["logprobs"])
