def assess_trust(confidence_result: dict, hallucination_result: dict) -> dict:
    evidence_level = confidence_result.get("level", "none")
    grounded = hallucination_result.get("grounded")

    if evidence_level == "none":
        return {
            "trust": "none",
            "reason": "No evidence was retrieved to support an answer.",
            "evidence_level": evidence_level,
            "grounded": grounded
        }

    if grounded is None:
        return {
            "trust": "unknown",
            "reason": "Groundedness could not be verified, so overall trust can't be confirmed.",
            "evidence_level": evidence_level,
            "grounded": grounded
        }

    if not grounded:
        return {
            "trust": "low",
            "reason": "The answer includes details not clearly supported by the retrieved evidence, regardless of how strong that evidence was.",
            "evidence_level": evidence_level,
            "grounded": grounded
        }

    return {
        "trust": evidence_level,
        "reason": f"The answer is faithful to the retrieved evidence, which was {evidence_level} confidence.",
        "evidence_level": evidence_level,
        "grounded": grounded
    }