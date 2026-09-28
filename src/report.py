"""PDF prediction report (fpdf2)."""
from __future__ import annotations

import re
from datetime import datetime


def _safe(s: str) -> str:
    """Core PDF fonts are latin-1 only; also break very long tokens so cells can wrap."""
    s = re.sub(r"(\S{60})", r"\1 ", str(s))
    return s.encode("latin-1", "replace").decode("latin-1")


def build_pdf(text: str, result: dict, real_df=None, fake_df=None, why: str = "") -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    nl = dict(new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Fake News Detection Report", **nl)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}", **nl)
    pdf.ln(4)

    is_fake = result["label"] == "FAKE"
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(200, 40, 40) if is_fake else pdf.set_text_color(30, 140, 70)
    pdf.cell(0, 10, f"Verdict: {result['label']}  ({result['confidence']:.1%} confidence)", **nl)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"P(REAL) = {result['prob_real']:.4f}    P(FAKE) = {result['prob_fake']:.4f}", **nl)
    pdf.ln(3)

    def section(title, body):
        pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 8, title, **nl)
        pdf.set_font("Helvetica", "", 10); pdf.multi_cell(0, 5.5, _safe(body), **nl); pdf.ln(2)

    excerpt = " ".join(text.split())
    section("Article excerpt", excerpt[:1500] + ("..." if len(excerpt) > 1500 else ""))
    if why:
        section("Why this prediction?", why.replace("**", ""))
    for title, df in (("Top words supporting REAL", real_df), ("Top words supporting FAKE", fake_df)):
        if df is not None and len(df):
            section(title, "\n".join(f"- {r.word}  ({r.score:+.4f})" for r in df.itertuples()))
    section("Disclaimer", "Automated model output based on writing patterns in the training "
            "data. It does not verify facts. Always cross-check with trusted sources.")
    return bytes(pdf.output())
