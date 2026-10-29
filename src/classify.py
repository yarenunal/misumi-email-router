from typing import Literal

import ollama
from pydantic import BaseModel, Field

from load_data import load_emails

MODEL = "gemma3:4b"


class Classification(BaseModel):
    """Modelin vermesi gereken cevabın kalıbı."""

    label: Literal["order", "quote", "return", "support", "other"]
    confidence: float = Field(ge=0, le=1)
    reason: str


SYSTEM_PROMPT = """You are an email triage assistant for an industrial components supplier.
Classify each inbound customer email into exactly one category:

- order: The customer places an order, sends a PO, approves a quote and asks to ship,
  or asks to reorder. They want goods to be shipped.
- quote: The customer asks for a price, lead time, quotation or RFQ, but has not
  committed to buy yet.
- return: The customer wants to send goods back, asks for an RMA, refund or credit,
  or reports wrong, damaged or defective items and wants them replaced or returned.
- support: The customer needs help or information: technical questions, order status
  or tracking, website or account problems, troubleshooting, and billing or account
  administration (invoice copies, questions or disputes about an invoice amount,
  address or tax form updates, account changes).
- other: Spam, marketing, job applications, auto-replies, phishing, internal or
  unrelated messages.

Rules for ambiguous cases:
- Approving a quote and asking to ship is an order.
- Asking for a price and saying they MIGHT order later is a quote.
- A problem report where the customer wants to return or get a refund is a return.
  A problem report where the customer only wants advice is support.
- Negotiating price or quantity BEFORE committing to buy is a quote.
- Only choose return if the customer explicitly mentions sending goods back, a refund,
  a credit, or a replacement. Do not assume intent that is not written in the email.
- Customer service and administrative requests (billing, accounts, addresses,
  documents) belong to support, not other. Use other only for messages that are not
  from a customer about doing business with us.

Respond with a JSON object with these fields:
- label: one of order, quote, return, support, other
- confidence: a number between 0 and 1 showing how sure you are
- reason: one short sentence explaining your choice
"""


def classify_email(subject, body):
    """Bir e-postayı modele gönder, Classification nesnesi döndür."""
    user_message = f"Subject: {subject}\n\nBody:\n{body}"

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=Classification.model_json_schema(),
        options={"temperature": 0},
    )

    return Classification.model_validate_json(response.message.content)


if __name__ == "__main__":
    emails = load_emails()

    # Sadece ilk 5 e-postayı dene (hepsini Adım 3'te çalıştıracağız)
    for email in emails[:5]:
        result = classify_email(email["subject"], email["body"])
        durum = "DOĞRU" if result.label == email["label"] else "YANLIŞ"
        print(f"[{durum}] {email['subject']}")
        print(f"   gerçek: {email['label']}  |  tahmin: {result.label}  |  güven: {result.confidence}")
        print(f"   gerekçe: {result.reason}\n")
