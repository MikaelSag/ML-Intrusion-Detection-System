
def tag_flow(flow: dict) -> list:
    tags = []
    if flow.get("dur", 1) < 0.5 and flow.get("spkts", 0) > 30:
        tags.append("possible-port-scan")
    if flow.get("rate", 0) > 40000:
        tags.append("possible-dos")
    if flow.get("sbytes", 0) > 10000 and flow.get("dur", 1) < 1:
        tags.append("high-bandwidth-burst")
    return tags


def infer_attack_label(tags: list) -> str:
    if "possible-dos" in tags:
        return "DoS"
    if "possible-port-scan" in tags:
        return "Reconnaissance"
    if "high-bandwidth-burst" in tags:
        return "Exfiltration"
    return "Unknown"
