"""Reading OpenStreetMap opening hours, conservatively.

The OSM ``opening_hours`` format is a small language — "Mo-Fr 09:00-17:00;
Sa 10:00-14:00; Su off", plus month ranges, public holidays, sunrise offsets
and more. This reads the common subset: day lists and ranges, one or more time
ranges, "off", and "24/7". Anything else (month or date ranges, holidays,
week numbers, comments) makes the whole value unreadable and ``parse`` returns
None — a schedule check that guesses at opening hours would flag good plans and
pass bad ones, so it only acts on hours it read for certain.
"""

from __future__ import annotations

import re

DAYS = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

#: Minutes since midnight, [start, end). An end past midnight is > 1440.
Interval = tuple[int, int]

_TIME = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\+?$")
_DAY_SPEC = re.compile(r"^(?:Mo|Tu|We|Th|Fr|Sa|Su)(?:-(?:Mo|Tu|We|Th|Fr|Sa|Su))?$")


def _days(spec: str) -> list[int] | None:
    """'Mo-Fr,Su' -> [0, 1, 2, 3, 4, 6]. None if it is not a plain day list."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part in ("PH", "SH"):  # holidays: not modelled, not needed to read the rest
            continue
        if not _DAY_SPEC.match(part):
            return None
        start, _, end = part.partition("-")
        a, b = DAYS.index(start), DAYS.index(end or start)
        out += list(range(a, b + 1)) if a <= b else [*range(a, 7), *range(0, b + 1)]
    return out


def _times(spec: str) -> list[Interval] | None:
    """'09:00-12:00,13:00-17:00' -> intervals. None if unreadable."""
    if spec in ("off", "closed"):
        return []
    out: list[Interval] = []
    for part in spec.split(","):
        m = _TIME.match(part.strip())
        if not m:
            return None
        h1, m1, h2, m2 = map(int, m.groups())
        start, end = h1 * 60 + m1, h2 * 60 + m2
        out.append((start, end if end > start else end + 24 * 60))
    return out


def parse(value: str | None) -> dict[int, list[Interval]] | None:
    """Weekday (0 = Monday) -> open intervals; [] means closed that day.

    Later rules override earlier ones for the days they name, as in OSM.
    """
    if not value:
        return None
    value = value.strip()
    if value == "24/7":
        return {d: [(0, 24 * 60)] for d in range(7)}

    week: dict[int, list[Interval]] = {}
    for rule in filter(None, (r.strip() for r in value.split(";"))):
        head, _, rest = rule.partition(" ")
        days = _days(head)
        if days is None:  # no day spec: the rule covers every day
            days, rest = list(range(7)), rule
        times = _times(rest.strip())
        if times is None or not days and head not in ("PH", "SH"):
            return None
        for d in days:
            week[d] = times
    return week or None


def open_at(week: dict[int, list[Interval]], day: int | None, minute: int) -> bool | None:
    """Is it open at `minute` past midnight on weekday `day`?

    With `day` unknown (a trip without dates), True if it is open at that time
    on *some* day it lists — so only "never open at that hour" is flagged.
    None if the hours do not cover that day at all.
    """
    days = [day] if day is not None else list(week)
    covered = [d for d in days if d in week]
    if not covered:
        return None
    return any(s <= minute < e for d in covered for s, e in week[d])


def clock(value: str) -> int | None:
    """'09:30' -> 570. None for slots like 'Evening'."""
    m = re.match(r"^(\d{1,2}):(\d{2})$", value.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


if __name__ == "__main__":
    week = parse("Mo-Su 09:00-17:00")
    assert week and open_at(week, None, 10 * 60) and not open_at(week, None, 18 * 60)
    week = parse("Tu-Su 10:00-18:00; Mo off")
    assert week and open_at(week, 0, 11 * 60) is False and open_at(week, 1, 11 * 60)
    assert parse("Mar 30-Sep 30: 08:30-19:15") is None  # months: not read, not guessed
    assert parse("Mo-Su,PH 10:00-18:00+")[6] == [(600, 1080)]
    assert parse("Fr-Sa 18:00-02:00")[4] == [(1080, 1560)]
    print("hours ok")
