from src.severity import classify_severity

THRESHOLD = 0.6

tests = [0.10, 0.69, 0.75, 0.85, 0.95]

for p in tests:
    print(p, "->", classify_severity(p, THRESHOLD))