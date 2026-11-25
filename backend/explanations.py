"""Plain-language explanations generated locally from model output only."""

from __future__ import annotations

from typing import Any, Dict, Optional


DISCLAIMER = (
    "This software is an automated research prototype, not a medical device or diagnosis. "
    "It does not replace review by a qualified clinician."
)


def local_explanation(
    result: Dict[str, Any],
    symptoms: Optional[str] = None,
    clinical_history: Optional[str] = None,
) -> Dict[str, Any]:
    prediction = result["prediction"]
    probability = float(result["probability"])
    confidence = float(result["confidence"])
    threshold = float(result["threshold"])
    normal_probability = float(result["probabilities"]["NORMAL"])
    pneumonia_probability = float(result["probabilities"]["PNEUMONIA"])
    margin = abs(probability - threshold)
    model = result.get("model", {})
    context_recorded = bool((symptoms or "").strip() or (clinical_history or "").strip())
    if prediction == "PNEUMONIA":
        summary = (
            "The model classified this chest X-ray as more consistent with the pneumonia class. "
            "This is an automated pattern classification and does not confirm pneumonia."
        )
        decision = (
            f"The model-estimated pneumonia score ({probability:.3f}) is at or above its "
            f"decision threshold ({threshold:.3f}), so the selected class is PNEUMONIA."
        )
        recommendation = (
            "Arrange timely review with a qualified healthcare professional, who can interpret "
            "the image alongside symptoms, examination, and other tests."
        )
    else:
        summary = (
            "The model classified this chest X-ray as more consistent with the normal class. "
            "A normal model result cannot rule out disease or replace clinical interpretation."
        )
        decision = (
            f"The model-estimated pneumonia score ({probability:.3f}) is below its decision "
            f"threshold ({threshold:.3f}), so the selected class is NORMAL."
        )
        recommendation = (
            "Continue to follow clinical advice. Seek prompt medical care if symptoms are severe, "
            "worsening, or otherwise concerning."
        )

    return {
        "summary": summary,
        "model_output": (
            f"Predicted class: {prediction}. Model confidence for that class: "
            f"{confidence * 100:.1f}%. Model-estimated pneumonia score: {probability * 100:.1f}%."
        ),
        "source": {
            "provider": "Local image classifier",
            "generation_method": "Deterministic explanation from classifier output",
            "gemini_used": False,
            "generative_ai_used": False,
        },
        "score_breakdown": {
            "normal_score": normal_probability,
            "pneumonia_score": pneumonia_probability,
            "predicted_class_confidence": confidence,
            "probability_interpretation": (
                "These are model scores, not calibrated clinical probabilities or guarantees "
                "of diagnostic accuracy."
            ),
        },
        "decision_detail": {
            "threshold": threshold,
            "rule": "PNEUMONIA is selected when the pneumonia score is greater than or equal to the threshold.",
            "explanation": decision,
            "distance_from_threshold": round(margin, 4),
            "distance_note": (
                "Distance from the model's classification cutoff only; it is not a measure of "
                "clinical certainty."
            ),
        },
        "clinical_context": {
            "recorded": context_recorded,
            "used_for_prediction": False,
            "note": (
                "Symptoms and history may be stored with the record, but this classifier does "
                "not use them to produce its image classification."
            ),
        },
        "scope": {
            "image_findings_generated": False,
            "finding_localization_available": False,
            "note": (
                "This model returns a whole-image class score. It does not identify specific "
                "anatomical findings, locate abnormalities, measure disease severity, or produce "
                "a radiologist-style report."
            ),
        },
        "technical_details": {
            "model_name": model.get("name", "Pneumonia image classifier"),
            "architecture": model.get("backbone", "Not available"),
            "input_size": model.get("img_size"),
            "decision_threshold": threshold,
        },
        "recommendation": recommendation,
        "health_improvements": (
            "This X-ray model cannot determine personal health improvements. Discuss prevention, "
            "smoking cessation, vaccination, and other relevant care with your clinician."
        ),
        "context_recorded": context_recorded,
        "disclaimer": DISCLAIMER,
    }
