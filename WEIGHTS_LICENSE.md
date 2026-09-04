# Model Weights License

The **model weights** distributed with Occular-OCR — including, without limitation, the text
detector, the text recognizer(s) (`svtr_lcnet` and `svtr_t`, for Russian/English and for the 12
Cyrillic-script languages), the language model(s), and the optional reading-order model — are
licensed under a **modified AI Pubs OpenRAIL-M** license. This applies to every such weight file
shipped with Occular-OCR or distributed for it from Hugging Face (`Shivin11/occular-ocr`), whether
downloaded automatically or placed in `weights/` by hand.

> **Exception — the page-orientation model.** The page-orientation weights
> (`orientation_orinet_fp32.onnx`) are **not** covered by this license. They are released under the
> **Apache License 2.0**, the same terms as the Occular-OCR source code — free for any use,
> including commercial, subject only to attribution. The OpenRAIL-M terms below do not apply to
> that file.

> The **source code** of Occular-OCR is separately licensed under the Apache License 2.0
> (see [`LICENSE`](LICENSE)). This file governs only the model weights.

## Free use

Use of the weights is **free** for:

- **Individuals** — personal, non-commercial use.
- **Researchers and academics.**
- **Self-employed individuals** without employees.
- **Non-profit organizations.**
- **Small organizations** — annual revenue under **20 000 000 ₽** **and** fewer than **8**
  employees.

Free use includes running the models, fine-tuning them, and shipping them inside your own products,
subject to the Use Restrictions below.

## Commercial license

Organizations that exceed the small-organization thresholds above (i.e. **20 000 000 ₽ or more in
annual revenue, or 8 or more employees**) must obtain a commercial license before using the weights
in production.

**Price: 300 000 ₽ per year, per organization.**

**Commercial licensing enquiries:**
- Email: **user26665@gmail.com**
- Telegram: **[@Bodhi_b](https://t.me/Bodhi_b)**

## Use restrictions (OpenRAIL-M)

Regardless of the tier above, you may **not** use the models to:

- generate or facilitate disinformation, or content intended to defame, harass, or deceive;
- violate applicable laws or the rights (including privacy) of others;
- discriminate against individuals or groups in a manner prohibited by law.

These restrictions must be passed on to any downstream users to whom you distribute the weights or
derivatives of them.

## Attribution

If you distribute the weights or a fine-tuned derivative, retain this license and credit
Occular-OCR.

---

*This is a plain-language summary of the applicable modified AI Pubs OpenRAIL-M terms. Consult the
full OpenRAIL-M text if you need the exact legal language.*
