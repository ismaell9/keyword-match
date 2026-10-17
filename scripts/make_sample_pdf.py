#!/usr/bin/env python3
"""Generate a small, valid sample PDF for the corpus — pure stdlib, no reportlab.

Why hand-roll a PDF: the corpus needs at least one PDF to exercise the pypdf
loader and the page-number citation path, and pulling in a heavy PDF *writer*
just to produce a fixture is not worth the dependency. This builds a minimal
multi-page PDF 1.4 with uncompressed content streams, which pypdf reads fine.

Run:  python scripts/make_sample_pdf.py
Out:  data/corpus/vendor-risk-summary.pdf
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "corpus" / "vendor-risk-summary.pdf"

PAGES = [
    [
        "Vendor Risk Summary",
        "",
        "This document summarises the risk assessment for third-party",
        "processors used by the Platform Engineering group. It is reviewed",
        "each quarter by the Security Office and the Data Protection Officer.",
        "",
        "Tiering",
        "Vendors are placed in one of three tiers. Tier 1 vendors process",
        "Restricted data and require an annual on-site assessment. Tier 2",
        "vendors process Internal data and require a security questionnaire",
        "every two years. Tier 3 vendors receive no company data and require",
        "only a signed terms of service.",
    ],
    [
        "Assessment findings",
        "",
        "The primary cloud hosting vendor is Tier 1 and holds a current",
        "independent audit report. No critical findings are open.",
        "",
        "The analytics vendor is Tier 2. Its questionnaire from the previous",
        "cycle lapsed in March, so a renewal is outstanding. Until the",
        "questionnaire is renewed, no new data may be shared with that vendor.",
        "",
        "Continuous monitoring",
        "Security ratings for all Tier 1 and Tier 2 vendors are monitored",
        "monthly. A drop in rating below the acceptable threshold opens an",
        "automatic review ticket assigned to the vendor owner.",
    ],
    [
        "Escalation and remediation",
        "",
        "A vendor with an open critical finding must produce a remediation",
        "plan within 10 business days. If the plan is rejected, the vendor is",
        "suspended: existing integrations are switched to read-only and new",
        "data flows are blocked.",
        "",
        "Termination of a vendor relationship requires confirmation that all",
        "company data has been deleted or returned, evidenced in writing, and",
        "recorded in the vendor register.",
    ],
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf() -> bytes:
    """Assemble a minimal PDF with one content stream per page."""
    objects: list[bytes] = []
    n_pages = len(PAGES)

    # Object numbering: 1=catalog, 2=pages, then 3..(2+n)=page objects,
    # then content streams, then the font.
    page_obj_ids = [3 + i for i in range(n_pages)]
    content_obj_ids = [3 + n_pages + i for i in range(n_pages)]
    font_id = 3 + 2 * n_pages

    def obj(num: int, body: bytes) -> None:
        while len(objects) < num:
            objects.append(b"")
        objects[num - 1] = f"{num} 0 obj\n".encode() + body + b"\nendobj\n"

    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    obj(1, f"<< /Type /Catalog /Pages 2 0 R >>".encode())
    obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())

    for i, lines in enumerate(PAGES):
        pid = page_obj_ids[i]
        cid = content_obj_ids[i]
        obj(
            pid,
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {cid} 0 R >>"
            ).encode(),
        )
        # Simple text layout: 14pt leading, starting near the top of the page.
        y = 720
        ops = ["BT", "/F1 11 Tf", "14 TL", f"1 0 0 1 72 {y} Tm"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        obj(cid, b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    obj(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for o in objects:
        offsets.append(len(out))
        out += o
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build_pdf()
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data)} bytes, {len(PAGES)} pages)")


if __name__ == "__main__":
    main()
