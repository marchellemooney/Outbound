"""
BDR Call Transcript Analysis — Mangomint
Analyzes 212 call transcripts for:
  1. Most common opening lines
  2. Objection patterns (type, frequency, position, rep response)
  3. Talk-time ratios by rep
  4. What top-converting calls have in common
"""

import json
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────
TRANSCRIPT_FILE = "mangomint_transcripts_212.json"
REPORT_FILE = "bdr_analysis_report.md"

KNOWN_REPS = {"Cylee Smart", "Alex Manso", "Sanaz Jahanmir", "Jo Galan"}

# Objection phrases — mapped to canonical category
OBJECTION_PATTERNS = {
    "Already has software / happy with current": [
        r"\b(happy with|love|like).{0,25}(vagaro|mindbody|boulevard|square|booker|fresha|zenoti|gloss|phorest|salon iris)\b",
        r"\b(already (use|using|have|on)|currently (use|using|on|with))\b",
        r"\bnot looking to (change|switch|move)\b",
        r"\bjust switched\b",
        r"\brecently (switched|moved|upgraded)\b",
    ],
    "Too busy / bad timing": [
        r"\b(too busy|really busy|swamped|not a good time|bad time|crazy (busy|time)|hectic)\b",
        r"\b(in the middle of|right in the)\b",
        r"\bcan('t| not) (talk|chat|do this) right now\b",
        r"\bgrand opening\b",
        r"\bjust opened\b",
        r"\bnew (location|salon|spa)\b",
    ],
    "Not interested": [
        r"\bnot (interested|looking)\b",
        r"\bno thank(s| you)\b",
        r"\bdon't (need|want|have time)\b",
        r"\btake (me off|us off).{0,15}(list|calling)\b",
        r"\bdo not (call|contact)\b",
        r"\bremove (me|us)\b",
    ],
    "Price / cost concern": [
        r"\b(too expensive|can't afford|out of (my |our )budget|cost(s)? (too much|a lot)|pricing concern)\b",
        r"\bwhat.{0,20}(cost|price|charge|fee)\b",
        r"\bhow much.{0,20}(is it|does it cost|per month)\b",
        r"\b(cheap|cheaper|less expensive|more affordable)\b",
        r"\b(budget|afford)\b",
    ],
    "Need to involve partner / decision maker": [
        r"\b(partner|husband|wife|spouse|co-owner|business partner).{0,30}(talk|discuss|decide|check|loop in)\b",
        r"\bnot (my|the sole) decision\b",
        r"\b(need to|have to|want to) (talk|discuss|check) with\b",
        r"\bother (partner|owner)\b",
    ],
    "Call me back later / follow up": [
        r"\b(call (me )?back|reach (me )?back|try (me )?(again|later))\b",
        r"\b(better time|good time).{0,20}(later|another day|next week|in a few)\b",
        r"\bin (a few|a couple of) (weeks?|months?|days?)\b",
        r"\bmaybe (later|another time|next)\b",
    ],
    "Contract / commitment concern": [
        r"\b(contract|lock(ed)? in|tied (up|down)|commitment|annual|year(-long)?)\b",
        r"\b(cancel|cancellation|exit|leave).{0,20}(anytime|easy|hard|penalty|fee)\b",
        r"\bmonth.to.month\b",
    ],
    "Already considering another option": [
        r"\b(looking at|checking out|comparing|evaluating|demoing).{0,25}(vagaro|mindbody|boulevard|square|booker|fresha|zenoti|gloss)\b",
        r"\b(another (option|solution|platform|demo)|few (options|demos))\b",
    ],
}

# Conversion detection — strong booking signals in last 40% of call
BOOKING_SIGNALS = [
    r"\b(monday|tuesday|wednesday|thursday|friday|tomorrow|next week).{0,50}(works|work|perfect|great|good|sounds good|that works)\b",
    r"\b(works|work|perfect|great|good|sounds good|that works).{0,50}(monday|tuesday|wednesday|thursday|friday|tomorrow)\b",
    r"\b\d{1,2}(am|pm|:\d{2}).{0,30}(works|perfect|great|good|sounds good)\b",
    r"\bsee you.{0,30}(monday|tuesday|wednesday|thursday|friday|tomorrow|then|at)\b",
    r"\blooking forward.{0,20}(to (it|that|our|the|chatting|talking|connecting))\b",
    r"\bi.ll (send|shoot).{0,30}(invite|link|calendar|email|over)\b",
    r"\b(best|good) email.{0,20}(for you|to send|to reach|I can|we can)\b",
    r"\bconfirm.{0,30}(time|appointment|meeting|demo)\b",
    r"\bset.{0,10}(up|a).{0,20}(time|call|demo|meeting)\b",
    r"\bschedule.{0,20}(a |the )?(call|demo|meeting|time)\b",
    r"\b(book|booked).{0,15}(for|a|the)\b",
    r"\bput.{0,10}(it |you )?(on|in).{0,10}(the |my )?(calendar|schedule)\b",
]

NOT_CONVERTED_SIGNALS = [
    r"\bnot interested\b",
    r"\bdo not (call|contact|reach out)\b",
    r"\btake (me|us) off.{0,15}list\b",
    r"\bremove (me|us)\b",
    r"\bno thank you\b",
]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    speaker: str
    timestamp: str
    text: str
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class Call:
    id: str
    index: int
    rep: Optional[str]
    turns: list
    converted: bool = False
    objections_found: list = field(default_factory=list)
    rep_word_count: int = 0
    prospect_word_count: int = 0

    @property
    def total_turns(self):
        return len(self.turns)

    @property
    def rep_talk_ratio(self):
        total = self.rep_word_count + self.prospect_word_count
        return self.rep_word_count / total if total > 0 else 0

    @property
    def prospect_talk_ratio(self):
        total = self.rep_word_count + self.prospect_word_count
        return self.prospect_word_count / total if total > 0 else 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def identify_rep(turns: list) -> Optional[str]:
    """Return the rep name from the turn list — the most frequent KNOWN_REPS speaker."""
    counts = Counter(t.speaker for t in turns if t.speaker in KNOWN_REPS)
    if counts:
        return counts.most_common(1)[0][0]
    # Fallback: the speaker with the most turns overall (heuristic)
    all_counts = Counter(t.speaker for t in turns)
    if all_counts:
        top = all_counts.most_common(1)[0][0]
        # Only return if they appear in > 30% of turns (reps dominate)
        if all_counts[top] / len(turns) > 0.3:
            return top
    return None


def is_converted(turns: list) -> bool:
    """Return True if the call ends with a demo/meeting booked."""
    last_portion = turns[max(0, int(len(turns) * 0.55)):]
    text = " ".join(t.text for t in last_portion).lower()
    full_text = " ".join(t.text for t in turns).lower()

    hard_no = any(re.search(p, full_text) for p in NOT_CONVERTED_SIGNALS)
    if hard_no:
        return False

    return any(re.search(p, text) for p in BOOKING_SIGNALS)


def get_first_rep_turn(turns: list, rep: str) -> Optional[Turn]:
    """Return the first substantive turn by the rep (>10 words)."""
    for t in turns:
        if t.speaker == rep and t.word_count > 10:
            return t
    for t in turns:
        if t.speaker == rep:
            return t
    return None


def normalize_opening(text: str) -> str:
    """Strip filler and reduce to first ~25 meaningful words for clustering."""
    text = text.lower().strip()
    # Remove leading filler words
    filler = r"^(hey|hi|hello|yeah|oh|okay|um|uh|so|well|yes|no|mhmm|right|alright|sure|great|good|perfect|awesome|absolutely|definitely|of course|sounds good)[,!.\s]+"
    for _ in range(6):
        text = re.sub(filler, "", text).strip()
    words = text.split()[:30]
    return " ".join(words)


def detect_objections(turns: list, rep: str) -> list:
    """Return list of (category, turn_index, turn_position_pct, prospect_text) for each objection."""
    found = []
    n = len(turns)
    for i, turn in enumerate(turns):
        if turn.speaker == rep:
            continue
        text = turn.text.lower()
        for category, patterns in OBJECTION_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                found.append({
                    "category": category,
                    "turn_index": i,
                    "position_pct": round(i / n * 100, 1),
                    "text": turn.text[:200],
                })
                break  # one objection category per turn
    return found


def talk_time(turns: list, rep: str):
    """Return (rep_words, prospect_words) tuple."""
    rep_w = sum(t.word_count for t in turns if t.speaker == rep)
    prospect_w = sum(t.word_count for t in turns if t.speaker != rep)
    return rep_w, prospect_w


# ── Main analysis ──────────────────────────────────────────────────────────────

def load_calls(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    calls = []
    for r in raw:
        turns = [Turn(speaker=t["speaker"], timestamp=t.get("timestamp", ""),
                      text=t["text"]) for t in r["transcript"]]
        rep = identify_rep(turns)
        rep_w, prospect_w = talk_time(turns, rep) if rep else (0, 0)
        objections = detect_objections(turns, rep) if rep else []
        converted = is_converted(turns)

        c = Call(
            id=r["id"],
            index=r["index"],
            rep=rep,
            turns=turns,
            converted=converted,
            objections_found=objections,
            rep_word_count=rep_w,
            prospect_word_count=prospect_w,
        )
        calls.append(c)
    return calls


# ── Section 1: Opening Lines ───────────────────────────────────────────────────

def analyze_openings(calls: list) -> dict:
    openings_by_rep = defaultdict(list)
    for c in calls:
        if not c.rep:
            continue
        first = get_first_rep_turn(c.turns, c.rep)
        if first:
            norm = normalize_opening(first.text)
            openings_by_rep[c.rep].append({
                "raw": first.text[:300],
                "normalized": norm,
                "converted": c.converted,
            })

    # Cluster by common opener themes
    theme_patterns = {
        "Compliment + competitor mention": [
            r"(love|love your|great|beautiful|amazing).{0,40}(brand|website|social|instagram|page|branding)",
            r"(came across|saw|found|noticed).{0,30}(website|instagram|profile|page)",
        ],
        "Direct competitor mention": [
            r"\b(vagaro|mindbody|boulevard|square|booker|fresha|zenoti|gloss|phorest)\b",
        ],
        "Salon/spa owner peer intro": [
            r"\b(salon owner|spa owner|work.{0,10}(salon|spa)|own a (salon|spa))\b",
        ],
        "Mangomint intro": [
            r"\b(mango ?mint|mangomint)\b",
        ],
        "Software/platform comparison pitch": [
            r"\b(all.in.one|platform|software|system|solution)\b",
            r"\b(next step|step up|upgrade|switch)\b",
        ],
        "Question opener (discovery first)": [
            r"^(how|what|when|where|are you|do you|have you|is this|can I).{0,60}\?",
            r"\bcurious.{0,30}(how|what|if)\b",
        ],
        "Callback / follow-up": [
            r"\b(follow(ing)? up|follow.up|called (before|earlier|last|yesterday|last week))\b",
            r"\b(we spoke|we talked|I called|I reached out)\b",
        ],
    }

    results = {}
    for rep, items in openings_by_rep.items():
        theme_counts = Counter()
        theme_conversion = defaultdict(list)
        for item in items:
            text = item["normalized"]
            matched = False
            for theme, pats in theme_patterns.items():
                if any(re.search(p, text, re.I) for p in pats):
                    theme_counts[theme] += 1
                    theme_conversion[theme].append(item["converted"])
                    matched = True
                    break
            if not matched:
                theme_counts["Other / unclear"] += 1
                theme_conversion["Other / unclear"].append(item["converted"])

        results[rep] = {
            "total": len(items),
            "themes": {
                theme: {
                    "count": count,
                    "conversion_rate": round(
                        sum(theme_conversion[theme]) / len(theme_conversion[theme]) * 100, 1
                    ) if theme_conversion[theme] else 0,
                }
                for theme, count in theme_counts.most_common()
            },
            "raw_sample": items[:5],
        }

    return results


# ── Section 2: Objection Patterns ─────────────────────────────────────────────

def analyze_objections(calls: list) -> dict:
    all_objections = []
    objections_by_rep = defaultdict(list)
    handled_converted = defaultdict(list)  # category -> [bool converted after?]

    for c in calls:
        for obj in c.objections_found:
            all_objections.append(obj)
            if c.rep:
                objections_by_rep[c.rep].append(obj)
            # Did call convert despite this objection?
            handled_converted[obj["category"]].append(c.converted)

    overall_freq = Counter(o["category"] for o in all_objections)
    avg_position = defaultdict(list)
    for o in all_objections:
        avg_position[o["category"]].append(o["position_pct"])

    return {
        "total_objections_detected": len(all_objections),
        "calls_with_objections": len([c for c in calls if c.objections_found]),
        "by_category": {
            cat: {
                "count": count,
                "avg_position_in_call_pct": round(
                    sum(avg_position[cat]) / len(avg_position[cat]), 1
                ),
                "conversion_rate_when_present": round(
                    sum(handled_converted[cat]) / len(handled_converted[cat]) * 100, 1
                ) if handled_converted[cat] else 0,
            }
            for cat, count in overall_freq.most_common()
        },
        "by_rep": {
            rep: Counter(o["category"] for o in objs).most_common(5)
            for rep, objs in objections_by_rep.items()
        },
    }


# ── Section 3: Talk-Time Ratios ────────────────────────────────────────────────

def analyze_talk_time(calls: list) -> dict:
    rep_stats = defaultdict(lambda: {
        "calls": 0,
        "total_rep_words": 0,
        "total_prospect_words": 0,
        "ratios": [],
        "converted_ratios": [],
        "not_converted_ratios": [],
    })

    for c in calls:
        if not c.rep:
            continue
        s = rep_stats[c.rep]
        s["calls"] += 1
        s["total_rep_words"] += c.rep_word_count
        s["total_prospect_words"] += c.prospect_word_count
        ratio = round(c.rep_talk_ratio * 100, 1)
        s["ratios"].append(ratio)
        if c.converted:
            s["converted_ratios"].append(ratio)
        else:
            s["not_converted_ratios"].append(ratio)

    summary = {}
    for rep, s in rep_stats.items():
        ratios = s["ratios"]
        sorted_r = sorted(ratios)
        summary[rep] = {
            "calls_analyzed": s["calls"],
            "avg_rep_talk_pct": round(sum(ratios) / len(ratios), 1) if ratios else 0,
            "median_rep_talk_pct": sorted_r[len(sorted_r) // 2] if sorted_r else 0,
            "converted_avg_rep_talk_pct": round(
                sum(s["converted_ratios"]) / len(s["converted_ratios"]), 1
            ) if s["converted_ratios"] else 0,
            "not_converted_avg_rep_talk_pct": round(
                sum(s["not_converted_ratios"]) / len(s["not_converted_ratios"]), 1
            ) if s["not_converted_ratios"] else 0,
            "calls_over_70pct_rep_talk": sum(1 for r in ratios if r > 70),
            "calls_under_50pct_rep_talk": sum(1 for r in ratios if r < 50),
        }
    return summary


# ── Section 4: Top-Converting Call Traits ─────────────────────────────────────

def analyze_conversion_traits(calls: list) -> dict:
    converted = [c for c in calls if c.converted]
    not_converted = [c for c in calls if not c.converted]

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    # Call length
    conv_lengths = [c.total_turns for c in converted]
    not_conv_lengths = [c.total_turns for c in not_converted]

    # Talk ratio
    conv_ratios = [round(c.rep_talk_ratio * 100, 1) for c in converted]
    not_conv_ratios = [round(c.rep_talk_ratio * 100, 1) for c in not_converted]

    # Objection handling (converted despite objections)
    conv_with_obj = [c for c in converted if c.objections_found]
    not_conv_with_obj = [c for c in not_converted if c.objections_found]

    # Prospect engagement: prospect turns as % of total
    conv_prospect_pct = [
        round(sum(1 for t in c.turns if t.speaker != c.rep) / c.total_turns * 100, 1)
        for c in converted if c.rep and c.total_turns > 0
    ]
    not_conv_prospect_pct = [
        round(sum(1 for t in c.turns if t.speaker != c.rep) / c.total_turns * 100, 1)
        for c in not_converted if c.rep and c.total_turns > 0
    ]

    # Keyword presence in top calls
    keywords = {
        "Competitor named (Vagaro/Mindbody/etc.)": [
            r"\b(vagaro|mindbody|boulevard|square|booker|fresha|zenoti|gloss|phorest|salon iris)\b"
        ],
        "Demo/screen share mentioned": [
            r"\b(demo|screen share|screen-share|walk you through|show you)\b"
        ],
        "Trial account offered": [
            r"\b(trial|free trial|trial account|try it|test it out)\b"
        ],
        "Pricing discussed": [
            r"\b(price|pricing|cost|per month|subscription|fee|how much)\b"
        ],
        "Client experience / retention mentioned": [
            r"\b(client (experience|retention|return|rebooking)|rebooking|retention|come back)\b"
        ],
        "Membership / packages mentioned": [
            r"\b(membership|memberships|package|packages|recurring)\b"
        ],
        "Pain point acknowledged": [
            r"\b(challenge|frustrat|struggle|pain|issue|problem|wish|could be better|improvement)\b"
        ],
        "Prospect asked a question": [
            r"\b(how do|does it|can I|is there|do you|what about|how does)\b"
        ],
    }

    def kw_rate(call_list, patterns):
        hits = 0
        for c in call_list:
            text = " ".join(t.text for t in c.turns).lower()
            if any(re.search(p, text) for p in patterns):
                hits += 1
        return round(hits / len(call_list) * 100, 1) if call_list else 0

    kw_comparison = {}
    for label, pats in keywords.items():
        kw_comparison[label] = {
            "converted_pct": kw_rate(converted, pats),
            "not_converted_pct": kw_rate(not_converted, pats) if not_converted else 0,
        }

    # Per-rep conversion rates
    rep_conversion = defaultdict(lambda: {"converted": 0, "total": 0})
    for c in calls:
        if c.rep:
            rep_conversion[c.rep]["total"] += 1
            if c.converted:
                rep_conversion[c.rep]["converted"] += 1

    return {
        "total_calls": len(calls),
        "converted": len(converted),
        "not_converted": len(not_converted),
        "conversion_rate_pct": round(len(converted) / len(calls) * 100, 1),
        "avg_call_length_turns": {
            "converted": avg(conv_lengths),
            "not_converted": avg(not_conv_lengths),
        },
        "avg_rep_talk_pct": {
            "converted": avg(conv_ratios),
            "not_converted": avg(not_conv_ratios),
        },
        "avg_prospect_turn_pct": {
            "converted": avg(conv_prospect_pct),
            "not_converted": avg(not_conv_prospect_pct),
        },
        "objection_handling": {
            "converted_calls_with_objections": len(conv_with_obj),
            "not_converted_calls_with_objections": len(not_conv_with_obj),
            "converted_objection_rate_pct": round(len(conv_with_obj) / len(converted) * 100, 1) if converted else 0,
            "not_converted_objection_rate_pct": round(len(not_conv_with_obj) / len(not_converted) * 100, 1) if not_converted else 0,
        },
        "keyword_presence": kw_comparison,
        "by_rep": {
            rep: {
                "calls": d["total"],
                "converted": d["converted"],
                "conversion_rate_pct": round(d["converted"] / d["total"] * 100, 1) if d["total"] else 0,
            }
            for rep, d in sorted(rep_conversion.items(), key=lambda x: -x[1]["total"])
        },
    }


# ── Report writer ──────────────────────────────────────────────────────────────

def bar(value, max_val=100, width=30, char="█"):
    filled = int(value / max_val * width) if max_val > 0 else 0
    return char * filled + "░" * (width - filled)


def write_report(calls, openings, objections, talk_time_data, conversion, path):
    lines = []
    a = lines.append

    a("# BDR Call Transcript Analysis — Mangomint")
    a(f"\n**Calls analyzed:** {len(calls)}  |  "
      f"**Reps:** {', '.join(sorted(set(c.rep for c in calls if c.rep)))}  |  "
      f"**Converted:** {conversion['converted']} ({conversion['conversion_rate_pct']}%)\n")
    a("---\n")

    # ── 1. Opening Lines ──────────────────────────────────────────────────────
    a("## 1. Opening Lines\n")
    a("The first substantive rep turn (>10 words) is categorized by theme below.\n")

    for rep, data in sorted(openings.items()):
        a(f"### {rep} ({data['total']} calls)\n")
        a("| Theme | Count | Conv. Rate |")
        a("|-------|-------|-----------|")
        for theme, stats in data["themes"].items():
            pct = stats['conversion_rate']
            a(f"| {theme} | {stats['count']} | {pct}% |")
        a("")
        a("**Sample opening lines:**\n")
        seen_samples = set()
        for item in data["raw_sample"]:
            norm = item["normalized"][:120]
            if norm not in seen_samples:
                seen_samples.add(norm)
                converted_label = "✓" if item["converted"] else "✗"
                a(f"> [{converted_label}] *\"{item['raw'][:220].strip()}...\"*\n")
        a("")

    # ── 2. Objection Patterns ─────────────────────────────────────────────────
    a("## 2. Objection Patterns\n")
    a(f"**{objections['total_objections_detected']} objections detected** across "
      f"**{objections['calls_with_objections']} calls** "
      f"({round(objections['calls_with_objections']/len(calls)*100,1)}% of calls).\n")

    a("### Objection Frequency & Conversion Impact\n")
    a("| Objection Type | Count | Avg Position | Conv. Rate When Present |")
    a("|----------------|-------|--------------|------------------------|")
    for cat, stats in objections["by_category"].items():
        pos = stats["avg_position_in_call_pct"]
        timing = "early" if pos < 33 else "mid" if pos < 66 else "late"
        a(f"| {cat} | {stats['count']} | {pos}% ({timing}) | {stats['conversion_rate_when_present']}% |")
    a("")

    a("### Objections by Rep\n")
    for rep, obj_list in objections["by_rep"].items():
        a(f"**{rep}:** " + ", ".join(f"{cat} ({cnt})" for cat, cnt in obj_list[:5]))
        a("")

    # ── 3. Talk-Time Ratios ───────────────────────────────────────────────────
    a("## 3. Talk-Time Ratios by Rep\n")
    a("> Industry benchmark: reps should aim for **~45–55% talk time** — enough to pitch, "
      "but leaving room for prospect to engage.\n")
    a("| Rep | Calls | Avg Rep Talk% | Median | Converted Avg | Not-Conv Avg | >70% (monologue risk) | <50% (low energy) |")
    a("|-----|-------|--------------|--------|---------------|--------------|----------------------|-------------------|")
    for rep, s in sorted(talk_time_data.items(), key=lambda x: -x[1]["calls_analyzed"]):
        avg = s["avg_rep_talk_pct"]
        a(f"| {rep} | {s['calls_analyzed']} | {avg}% {bar(avg, width=15)} | "
          f"{s['median_rep_talk_pct']}% | "
          f"{s['converted_avg_rep_talk_pct']}% | "
          f"{s['not_converted_avg_rep_talk_pct']}% | "
          f"{s['calls_over_70pct_rep_talk']} | "
          f"{s['calls_under_50pct_rep_talk']} |")
    a("")

    # ── 4. Top-Converting Call Traits ─────────────────────────────────────────
    a("## 4. What Top-Converting Calls Have in Common\n")

    a("### Conversion Rate by Rep\n")
    a("| Rep | Calls | Converted | Rate |")
    a("|-----|-------|-----------|------|")
    for rep, d in conversion["by_rep"].items():
        a(f"| {rep} | {d['calls']} | {d['converted']} | {d['conversion_rate_pct']}% |")
    a("")

    a("### Call Characteristics: Converted vs. Not Converted\n")
    a("| Metric | Converted | Not Converted | Delta |")
    a("|--------|-----------|---------------|-------|")

    cl = conversion["avg_call_length_turns"]
    delta_len = round(cl["converted"] - cl["not_converted"], 1)
    a(f"| Avg call length (turns) | {cl['converted']} | {cl['not_converted']} | {delta_len:+.1f} |")

    rt = conversion["avg_rep_talk_pct"]
    delta_rt = round(rt["converted"] - rt["not_converted"], 1)
    a(f"| Avg rep talk % | {rt['converted']}% | {rt['not_converted']}% | {delta_rt:+.1f}pp |")

    pp = conversion["avg_prospect_turn_pct"]
    delta_pp = round(pp["converted"] - pp["not_converted"], 1)
    a(f"| Avg prospect turn % | {pp['converted']}% | {pp['not_converted']}% | {delta_pp:+.1f}pp |")

    oh = conversion["objection_handling"]
    a(f"| Calls with objections | {oh['converted_objection_rate_pct']}% | "
      f"{oh['not_converted_objection_rate_pct']}% | "
      f"{round(oh['converted_objection_rate_pct'] - oh['not_converted_objection_rate_pct'], 1):+.1f}pp |")
    a("")

    a("### Keyword Presence: Converted vs. Not Converted\n")
    a("| Topic Mentioned | Converted % | Not Converted % | Lift |")
    a("|-----------------|------------|-----------------|------|")
    for label, d in sorted(conversion["keyword_presence"].items(),
                           key=lambda x: -(x[1]["converted_pct"] - x[1]["not_converted_pct"])):
        lift = round(d["converted_pct"] - d["not_converted_pct"], 1)
        sign = "+" if lift >= 0 else ""
        a(f"| {label} | {d['converted_pct']}% | {d['not_converted_pct']}% | {sign}{lift}pp |")
    a("")

    a("### Key Observations\n")

    # Auto-generate observations from data
    obs = []

    # Talk time
    for rep, s in talk_time_data.items():
        avg = s["avg_rep_talk_pct"]
        conv_avg = s["converted_avg_rep_talk_pct"]
        not_conv_avg = s["not_converted_avg_rep_talk_pct"]
        if s["calls_analyzed"] >= 5:
            if conv_avg < not_conv_avg - 3:
                obs.append(
                    f"- **{rep}** converts at a higher rate when talking *less* — "
                    f"converted calls avg **{conv_avg}%** rep talk vs **{not_conv_avg}%** on non-converted calls. "
                    f"More prospect air time correlates with booking."
                )
            elif not_conv_avg < conv_avg - 3:
                obs.append(
                    f"- **{rep}** shows higher conversion when talking *more* — "
                    f"converted calls avg **{conv_avg}%** vs **{not_conv_avg}%** on non-converted. "
                    f"This rep's energy and pacing may drive engagement."
                )

    # Call length
    cl = conversion["avg_call_length_turns"]
    if cl["converted"] > cl["not_converted"] + 5:
        obs.append(
            f"- **Converted calls are longer** ({cl['converted']} avg turns vs "
            f"{cl['not_converted']} for not-converted). Prospects who stay on the line longer are more likely to book."
        )
    elif cl["not_converted"] > cl["converted"] + 5:
        obs.append(
            f"- **Non-converted calls drag longer** ({cl['not_converted']} avg turns vs "
            f"{cl['converted']} for converted). Reps may be chasing reluctant prospects."
        )

    # Objection handling
    oh = conversion["objection_handling"]
    if oh["converted_objection_rate_pct"] < oh["not_converted_objection_rate_pct"] - 10:
        obs.append(
            f"- **Fewer objections on converted calls** ({oh['converted_objection_rate_pct']}% "
            f"vs {oh['not_converted_objection_rate_pct']}%). Strong openers that qualify prospects early "
            f"reduce downstream resistance."
        )
    elif oh["converted_objection_rate_pct"] > oh["not_converted_objection_rate_pct"]:
        obs.append(
            f"- **Reps convert even when objections are present** "
            f"({oh['converted_objection_rate_pct']}% of converted calls had detectable objections). "
            f"The ability to handle objections and still book is a differentiator."
        )

    # Top keywords
    kw = conversion["keyword_presence"]
    top_kw = sorted(kw.items(), key=lambda x: -(x[1]["converted_pct"] - x[1]["not_converted_pct"]))[:3]
    for label, d in top_kw:
        lift = round(d["converted_pct"] - d["not_converted_pct"], 1)
        if lift > 5:
            obs.append(
                f"- **\"{label}\"** appears in **{d['converted_pct']}%** of converted calls vs "
                f"**{d['not_converted_pct']}%** of non-converted (+{lift}pp). "
                f"This topic correlates strongly with booking."
            )

    if not obs:
        obs.append("- No statistically strong differentiators detected — sample may be too small for robust signal.")

    for o in obs:
        a(o)
    a("")

    # ── 5. Coaching Recommendations ───────────────────────────────────────────
    a("## 5. Coaching Recommendations\n")

    recs = []

    # Talk ratio coaching
    for rep, s in talk_time_data.items():
        avg = s["avg_rep_talk_pct"]
        monologue_risk = s["calls_over_70pct_rep_talk"]
        if avg > 65:
            recs.append(
                f"- **{rep} — Lower talk time:** Averaging **{avg}% rep talk**. "
                f"{monologue_risk} calls exceeded 70%. "
                f"Coach to pause after pitching and ask discovery questions rather than filling silence."
            )
        elif avg < 40:
            recs.append(
                f"- **{rep} — Increase confidence/pacing:** Averaging only **{avg}% rep talk**. "
                f"Rep may be losing the frame — ensure they're driving the call with clear value propositions."
            )

    # Objection-specific coaching
    for cat, stats in objections["by_category"].items():
        if stats["count"] >= 5 and stats["conversion_rate_when_present"] < 70:
            recs.append(
                f"- **Handle \"{cat}\" better:** Appears {stats['count']} times, "
                f"converting only **{stats['conversion_rate_when_present']}%** of the time. "
                f"Build a specific reframe script for this objection."
            )

    # Opening line coaching
    for rep, data in openings.items():
        themes = data["themes"]
        low_conv = {t: s for t, s in themes.items() if s["count"] >= 3 and s["conversion_rate"] < 70}
        for theme, s in low_conv.items():
            recs.append(
                f"- **{rep} — Rework \"{theme}\" opener:** "
                f"{s['count']} calls use this approach with only **{s['conversion_rate']}%** conversion. "
                f"Test a curiosity/question-led opener instead."
            )

    if not recs:
        recs.append("- Overall performance is strong. Focus on replicating top rep patterns across the team.")

    for r in recs:
        a(r)
    a("")

    a("---")
    a("*Analysis generated from 212 Mangomint BDR call transcripts. "
      "Conversion defined as: specific demo/meeting day+time confirmed or email captured for booking in the last 40% of the call.*")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading transcripts...")
    calls = load_calls(TRANSCRIPT_FILE)
    print(f"  Loaded {len(calls)} calls")

    print("Analyzing opening lines...")
    openings = analyze_openings(calls)

    print("Analyzing objection patterns...")
    objections = analyze_objections(calls)

    print("Analyzing talk-time ratios...")
    talk_time_data = analyze_talk_time(calls)

    print("Analyzing conversion traits...")
    conversion = analyze_conversion_traits(calls)

    print("Writing report...")
    write_report(calls, openings, objections, talk_time_data, conversion, REPORT_FILE)

    # Quick summary to stdout
    print("\n── Quick Summary ──────────────────────────────────────────────")
    print(f"  Calls: {len(calls)} | Converted: {conversion['converted']} ({conversion['conversion_rate_pct']}%)")
    print(f"  Objections detected: {objections['total_objections_detected']} across {objections['calls_with_objections']} calls")
    print(f"\n  Talk-time by rep:")
    for rep, s in sorted(talk_time_data.items(), key=lambda x: -x[1]["calls_analyzed"]):
        print(f"    {rep}: {s['avg_rep_talk_pct']}% avg rep talk ({s['calls_analyzed']} calls)")
    print(f"\n  Top objection: {list(objections['by_category'].keys())[0]}")
    print(f"\n  Full report: {REPORT_FILE}")
