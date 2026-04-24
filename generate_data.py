#!/usr/bin/env python3
"""
generate_data.py — Synthetic PII Dataset Generator
====================================================
Generates a realistic-looking synthetic dataset of 300,000 text rows for use
as input to the OpenAI Privacy Filter demo.  All data is clearly prefixed with
"[SYNTHETIC DATA]" and contains no real personal information.

Dataset composition
-------------------
- 70 % of rows contain one or more PII types drawn from all eight categories
  supported by the OpenAI Privacy Filter model:
    private_person, private_address, private_email, private_phone,
    private_url, private_date, account_number, secret
- 30 % of rows contain no PII (benign text), giving the model something to
  correctly leave untouched.

Output
------
synthetic_data.csv   — two columns: id (int), text (str)

Usage
-----
    python generate_data.py

Dependencies
------------
    pip install pandas faker
"""

import pandas as pd
from faker import Faker
import random

def main():
    fake = Faker()

    # Templates with PII (mix of types)
    templates_with_pii = [
        "Hello, my name is {private_person} and I live at {private_address}. You can email me at {private_email} or call {private_phone}.",
        "My account number is {account_number} and I was born on {private_date}. Visit my site at {private_url}.",
        "For security, my secret code is {secret}. Contact me at {private_email} or {private_phone}.",
        "Hi {private_person}, your order from {private_date} is ready. Tracking at {private_url}, account {account_number}.",
        "Address: {private_address}. Phone: {private_phone}. Email: {private_email}. Secret: {secret}.",
        "Born on {private_date}, account {account_number}, email {private_email}, phone {private_phone}.",
        "Name: {private_person}, URL: {private_url}, secret {secret}, date {private_date}.",
        "Full details: {private_person} at {private_address}, {private_email}, {private_phone}, {account_number}, {private_date}, {private_url}, {secret}."
    ]

    # Templates without PII
    templates_no_pii = [
        "This is a simple message without any personal information.",
        "The weather is nice today in the city.",
        "I enjoy reading books and watching movies.",
        "Technology is advancing rapidly these days.",
        "Let's discuss the project timeline tomorrow.",
        "The meeting was productive and informative.",
        "I prefer coffee over tea in the morning.",
        "Music helps me relax after a long day.",
        "Traveling to new places is always exciting.",
        "Cooking is one of my favorite hobbies."
    ]

    def generate_text(with_pii=True):
        if with_pii:
            template = random.choice(templates_with_pii)
            text = template.format(
                private_person=fake.name(),
                private_address=fake.address().replace('\n', ', '),
                private_email=fake.email(),
                private_phone=fake.phone_number(),
                private_date=fake.date_of_birth().strftime('%B %d, %Y'),
                private_url=fake.url(),
                account_number=fake.iban()[:12],  # Shortened account number
                secret=fake.password(length=8)
            )
        else:
            text = random.choice(templates_no_pii)
        return f"[SYNTHETIC DATA] {text}"

    # Generate 300,000 rows (70% with PII, 30% without)
    num_rows = 300000
    data = []
    print("Generating synthetic data...")
    for i in range(num_rows):
        with_pii = random.random() < 0.7
        text = generate_text(with_pii)
        data.append({'id': i+1, 'text': text})
        if (i+1) % 50000 == 0:
            print(f"Generated {i+1} rows...")

    df = pd.DataFrame(data)
    df.to_csv('synthetic_data.csv', index=False)
    print("Generated synthetic_data.csv with 300,000 rows")

if __name__ == "__main__":
    main()
