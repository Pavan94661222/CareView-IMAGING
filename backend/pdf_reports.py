"""Dependency-free generation of a concise printable analysis PDF."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List


def _pdf_string(value: str) -> str:
    normalized = value.encode("ascii", "replace").decode("ascii")
    return normalized.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _report_lines(record: Dict[str, Any]) -> List[str]:
    result = record["result"]
    patient_name = record.get("patient_name") or "Not provided"
    patient_code = record.get("patient_code") or "Not provided"
    explanation = result.get("explanation", {})
    scores = explanation.get("score_breakdown", {})
    decision = explanation.get("decision_detail", {})
    source = explanation.get("source", {})
    context = explanation.get("clinical_context", {})
    scope = explanation.get("scope", {})
    technical = explanation.get("technical_details", {})
    lines = [
        "CLINICAL IMAGE ANALYSIS REPORT",
        "",
        f"Analysis ID: {record['id']}",
        f"Created (UTC): {record['created_at']}",
        f"Patient: {patient_name}",
        f"Patient ID: {patient_code}",
        f"Age: {record.get('age') if record.get('age') is not None else 'Not provided'}",
        f"Sex: {record.get('sex') or 'Not provided'}",
        f"Image file: {record['image_filename']}",
        f"Image dimensions: {record['image_width']} x {record['image_height']}",
        "",
        f"Model classification: {result.get('prediction', 'Unavailable')}",
        f"Model confidence: {float(result.get('confidence', 0)) * 100:.1f}%",
        f"Model-estimated pneumonia score: {float(result.get('probability', 0)) * 100:.1f}%",
        f"Normal class score: {float(scores.get('normal_score', result.get('probabilities', {}).get('NORMAL', 0))) * 100:.1f}%",
        f"Pneumonia class score: {float(scores.get('pneumonia_score', result.get('probability', 0))) * 100:.1f}%",
        f"Decision threshold: {float(result.get('threshold', 0)):.3f}",
        "",
        "Explanation:",
        explanation.get("summary", "No explanation available."),
        explanation.get("model_output", ""),
        decision.get("explanation", ""),
        decision.get("distance_note", ""),
        "",
        f"Explanation source: {source.get('provider', 'Local image classifier')}",
        f"Generation method: {source.get('generation_method', 'Deterministic local model output')}",
        f"Gemini API used: {'Yes' if source.get('gemini_used') else 'No'}",
        f"Model: {technical.get('model_name', result.get('model', {}).get('name', 'Not available'))}",
        f"Architecture: {technical.get('architecture', result.get('model', {}).get('backbone', 'Not available'))}",
        "",
        "Clinical context:",
        context.get(
            "note",
            "Patient context is stored with the record and is not used by the image classifier.",
        ),
        "",
        "Model scope:",
        scope.get(
            "note",
            "Whole-image class score only; no localized findings or radiologist report are generated.",
        ),
        "",
        "Recommendation:",
        explanation.get("recommendation", "Consult a qualified healthcare professional."),
        explanation.get("health_improvements", ""),
        "",
        f"Symptoms: {record.get('symptoms') or 'Not provided'}",
        f"Clinical history: {record.get('clinical_history') or 'Not provided'}",
        f"Notes: {record.get('notes') or 'Not provided'}",
        "",
        "IMPORTANT: This automated research prototype is not a medical device or diagnosis.",
        "It does not replace a radiologist or other qualified clinician.",
        "Model confidence is not a guarantee of accuracy; interpret with clinical context.",
    ]
    wrapped: List[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=88, break_long_words=True, break_on_hyphens=False))
    return wrapped


def build_analysis_pdf(record: Dict[str, Any]) -> bytes:
    lines = _report_lines(record)
    pages = [lines[index : index + 48] for index in range(0, len(lines), 48)] or [[]]
    page_count = len(pages)
    font_number = 3 + page_count * 2
    objects = [b""] * (font_number + 1)
    page_references = []
    for index, page_lines in enumerate(pages):
        page_number = 3 + index * 2
        content_number = page_number + 1
        page_references.append(f"{page_number} 0 R".encode("ascii"))
        commands = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
        for line_index, line in enumerate(page_lines):
            if line_index:
                commands.append("T*")
            commands.append(f"({_pdf_string(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        objects[page_number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            + f"/Resources << /Font << /F1 {font_number} 0 R >> >> ".encode("ascii")
            + f"/Contents {content_number} 0 R >>".encode("ascii")
        )
        objects[content_number] = (
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        b"<< /Type /Pages /Kids ["
        + b" ".join(page_references)
        + b"] /Count "
        + str(page_count).encode("ascii")
        + b" >>"
    )
    objects[font_number] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects[1:], start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)
