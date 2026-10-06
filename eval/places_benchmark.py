"""How good is the place verifier itself?

    uv run python -m eval.places_benchmark

The verifier is the project's hallucination check, so its own error rate has to
be known. This is a small labelled set of real places and invented ones, run
through the live adapter.

**Precision is the number that matters.** A false positive means the system
confirms a place that does not exist, which is worse than confirming nothing:
it launders a hallucination as verified fact. A false negative only costs a
confirmation the plan never claimed to have.

Honest caveat: the matching thresholds in `adapters/places.py` were tuned partly
against this set, so a perfect score here is a sanity check, not an unbiased
estimate. It is 31 cases across two cities and two countries — treat it as
"no known failure mode", not "solved".
"""

from __future__ import annotations

from backend.adapters.places import PlacesAdapter

#: (place name as a model might write it, does it really exist there)
BENCHMARK: dict[str, tuple[tuple[str, bool], ...]] = {
    "Lisbon": (
        ("Jeronimos Monastery", True),
        ("Belém Tower", True),
        ("Castelo de Sao Jorge", True),
        ("Time Out Market Lisboa", True),
        ("National Tile Museum (Museu Nacional do Azulejo)", True),
        ("Praça do Comércio", True),
        ("Museum of Imaginary Tiles", False),
        ("Palacio do Nada", False),
        ("Temple of the Silver Fox", False),
        ("Grand Lisbon Aquarium of Mars", False),
    ),
    "Kyoto": (
        ("Fushimi Inari Taisha", True),
        ("Kinkaku-ji (Golden Pavilion)", True),
        ("Nijo Castle", True),
        ("Nishiki Market", True),
        ("Kiyomizu-dera", True),
        ("Philosopher's Path", True),
        ("Kyoto Museum of Invented History", False),
        ("Temple of the Eternal Noodle", False),
    ),
    # Country-level destinations: real places hundreds of km apart, and
    # invented ones that a country-wide search has more chances to mismatch.
    "Japan": (
        ("Senso-ji Temple", True),
        ("Fushimi Inari Taisha", True),
        ("Osaka Castle", True),
        ("Itsukushima Shrine", True),
        ("Hiroshima Peace Memorial Park", True),
        ("Osaka Museum of Floating Lanterns", False),
        ("Hokkaido Glass Volcano Park", False),
    ),
    "Italy": (
        ("Colosseum", True),
        ("Uffizi Gallery", True),
        ("Rialto Bridge", True),
        ("Leaning Tower of Pisa", True),
        ("Basilica of the Silent Moon", False),
        ("Venice Museum of Paper Boats", False),
    ),
}


def main() -> int:
    adapter = PlacesAdapter()
    if not adapter.live:
        print("GEOAPIFY_API_KEY is not set — the verifier cannot be benchmarked.")
        return 1

    tp = fp = tn = fn = 0
    misses: list[str] = []
    false_positives: list[str] = []

    for city, items in BENCHMARK.items():
        confirmed = adapter.fetch(
            destination=city, names=[name for name, _ in items]
        ).data["confirmed"]

        for name, real in items:
            matched = confirmed[name]
            if real and matched:
                tp += 1
            elif real:
                fn += 1
                misses.append(f"{city}: {name}")
            elif matched:
                fp += 1
                false_positives.append(f"{city}: {name} -> {matched}")
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0

    print(f"  real places confirmed   {tp}/{tp + fn}   (recall {recall:.0%})")
    print(f"  invented ones rejected  {tn}/{tn + fp}")
    print(f"  precision               {precision:.0%}   "
          f"{'ok' if precision == 1.0 else 'REGRESSION — a fake was confirmed'}")

    if false_positives:
        print("\n  confirmed something invented:")
        print("\n".join(f"    {f}" for f in false_positives))
    if misses:
        print("\n  could not confirm (real, costs a confirmation only):")
        print("\n".join(f"    {m}" for m in misses))

    return 0 if precision == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
