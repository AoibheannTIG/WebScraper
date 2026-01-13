import feedparser
import os
import json
import time
from google import genai
from datetime import datetime, timedelta
from dateutil import parser

# --- CONFIGURATION ---

# 1. Set your Gemini API Key here
# Get one at: https://aistudio.google.com/app/apikey
API_KEY = os.getenv("API_KEY", "your-api-key-here")  # Replace with your actual key or set env var
genai_client = genai.Client(api_key=API_KEY)

# 2. Define the High-Signal Source List
# We focus on sources that discuss 'breakthroughs'.
RSS_FEEDS = [
    # ==================================================
    # Layer 3 — Elite Journals (slow, deep, decisive)
    # ==================================================

    "https://jmlr.org/rss.xml",
    # Journal of Machine Learning Research (core ML algorithms)

    "https://cacm.acm.org/feed/",
    # Communications of the ACM (algorithmic breakthroughs, broad impact)

    "https://www.siam.org/rss/journals/sicomp.xml",
    # SIAM Journal on Computing (algorithms + complexity theory)

    "https://www.sciencedirect.com/rss/journal/00043702",
    # Artificial Intelligence (Elsevier – foundational AI)

    "https://rss.sciencedirect.com/publication/science/08936080",
    # Neural Networks (learning algorithms, theory-adjacent)


    # ==================================================
    # Layer 4 — Research Lab Blogs (early, explained)
    # ==================================================

    "https://news.mit.edu/rss/topic/computer-science-and-artificial-intelligence",
    # MIT News (algorithms, theory, optimization, foundational AI)

    "https://research.google/blog/rss/",
    # Google Research (algorithms, theory, efficiency)

    "https://deepmind.google/discover/blog/rss.xml",
    # DeepMind (learning algorithms, theoretical advances)

    "https://www.microsoft.com/en-us/research/feed/",
    # Microsoft Research (optimization, theory, systems)

    "https://ai.meta.com/blog/rss/",
    # Meta AI (representation learning, algorithmic ML)

    "https://www.ibm.com/blogs/research/feed/",
    # IBM Research (algorithms + applied theory)

    "https://engineering.nvidia.com/rss.xml",
    # NVIDIA Research & Engineering (algorithm–hardware boundary)


    # ==================================================
    # Layer 5 — Explanatory Science Media (sense-making)
    # ==================================================

    "https://api.quantamagazine.org/feed/",
    # Quanta Magazine (best math + CS explanations)

    "https://www.technologyreview.com/feed/",
    # MIT Technology Review (algorithmic + strategic framing)

    "https://www.nature.com/subjects/machine-learning.rss",
    # Nature Machine Intelligence

    "https://www.science.org/rss/computers.xml",
    # Science Magazine – computing & algorithms


    # ==================================================
    # Layer 6 — Curated / Community Signal
    # ==================================================

    "https://news.ycombinator.com/rss",
    # Hacker News (early social signal for real breakthroughs)

    "https://thegradient.pub/rss/",
    # The Gradient (theory-aware ML essays)

    "https://www.deeplearning.ai/thebatch/feed/",
    # The Batch (Jack Clark – strong signal, low noise)

    "https://importai.substack.com/feed",
    # Import AI (deep research synthesis)

    "https://distill.pub/rss.xml",
    # Distill (rare, but extremely high-quality)
]

# 3. Keywords to pre-filter noise (must match at least one to even send to AI)
KEYWORDS = [

    # ==================================================
    # Novelty / Discovery Language (VERY IMPORTANT)
    # ==================================================
    "new",
    "novel",
    "new way",
    "new method",
    "new approach",
    "new framework",
    "new technique",
    "previously unknown",
    "first time",
    "for the first time",
    "we propose",
    "we introduce",
    "we present",
    "we develop",
    "we show",
    "we demonstrate",
    "we prove",
    "we uncover",
    "we discover",
    "we identify",


    # ==================================================
    # Algorithm / Method Signals (CORE)
    # ==================================================
    "algorithm",
    "algorithmic",
    "learning algorithm",
    "training algorithm",
    "inference algorithm",
    "optimization algorithm",
    "search algorithm",
    "sampling algorithm",
    "approximation algorithm",
    "iterative method",
    "probabilistic method",
    "deterministic method",
    "computational method",
    "numerical method",


    # ==================================================
    # Efficiency / Constraint-Removal (TIG GOLD)
    # ==================================================
    "efficient",
    "more efficient",
    "efficiency",
    "scalable",
    "scaling",
    "reduces",
    "reducing",
    "improves",
    "improvement",
    "speedup",
    "faster",
    "lower cost",
    "less compute",
    "compute-efficient",
    "data-efficient",
    "memory-efficient",
    "sublinear",
    "linear time",
    "polynomial time",


    # ==================================================
    # Enabling / Unlocking Language (VERY HIGH SIGNAL)
    # ==================================================
    "enables",
    "enabling",
    "unlock",
    "unlocks",
    "makes it possible",
    "allows",
    "permits",
    "opens the door",
    "removes the need",
    "without requiring",
    "previously infeasible",
    "fundamental limitation",
    "bottleneck",

]

# 4. TIG Context for the AI
TIG_CONTEXT = """
ROLE
You are an analyst writing daily X posts for The Innovation Game (TIG) about *algorithmic breakthroughs* (the kind that change asymptotics, guarantees, or compute/data efficiency).

WHAT COUNTS AS A "BREAKTHROUGH" (HIGH SIGNAL)
- New algorithmic technique or framework (not a product feature).
- Better theoretical guarantees: approximation ratio, convergence rate, bounds, correctness.
- Better complexity: time/memory/communication (e.g., O(n^2)->O(n log n), near-linear, sublinear, FPT).
- Meaningful compute/data savings in training or inference *from an algorithmic change* (optimizer, scheduling, sparsity, routing, quantization with theory/ablations).
- Strong evidence: proofs, ablations, benchmark suite, reproducible comparisons.

WHAT DOES NOT COUNT (LOW SIGNAL)
- Model releases, partnerships, fundraising, “state-of-the-art” without algorithmic detail.
- Generic “efficiency improvements” with only hardware/infra changes.
- Opinion/policy/market commentary.
- Pure applications with no algorithmic novelty.

WHAT TIG IS
- The Innovation Game (TIG) is a decentralized protocol that incentivizes algorithmic breakthroughs.
- It creates a market where Innovators submit algorithms for important problems (e.g., Knapsack, SAT, VRP, neural network optimization, hypergraph partitioning, vector search, etc.).
- Benchmarkers (miners) compete by running standardized benchmarks and adopting the best-performing algorithms.
- Incentives:
  - Innovators are rewarded when their algorithm is adopted by benchmarkers.
  - Benchmarkers are rewarded for finding/adopting the most efficient algorithms.

CORE NARRATIVE (THE “WHY”)
- Algorithms underpin entire industries: they define what is feasible, what is cheap, and what scales.
- Algorithmic improvements can rival or exceed hardware gains (time, memory, data efficiency).
- Historically, many foundational algorithms were open and widely shared; today, algorithmic advantages are increasingly privatized (closed models, proprietary optimizers, internal infrastructure).
- TIG is part of the solution: an economic marketplace that rewards open, benchmarked algorithmic progress.




WHAT TO EXTRACT FROM AN ARTICLE
1) Algorithmic novelty (1–2 sentences):
   - Technique name + what it changes (complexity / guarantee / convergence / sample efficiency).
2) Constraint removed (1 sentence):
   - What previously limited performance? (scaling, memory, worst-case, stability, discrete structure, etc.)
3) Why it matters economically (1 sentence):
   - Cheaper compute, faster deployment, bigger feasible problem sizes, improved reliability.
4) TIG mapping (1–2 sentences):
   - Explain how TIG links to the article. EG: importance of algorithms, how TIG can help, etc.

OUTPUT (FOR X)
- A) One-sentence headline (plain English; no hype).
- B) 3 bullets: (Novelty) / (Constraint removed) / (Economic why).
- C) 1–2 sentences connecting to TIG (market adoption + benchmarking).
- If not genuinely algorithmic: say “Not an algorithmic breakthrough” and return no tweets.
STYLE
- Punchy, technical but accessible. Avoid buzzwords. Prefer concrete numbers/claims if present.
"""



import json
import re

def _extract_json_object(text: str):
    """
    Best-effort extraction of the first top-level JSON object from a string.
    Handles models that sometimes prepend/append text.
    """
    if not text:
        return None
    # Fast path
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except:
            pass

    # Search for a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except:
        return None

def _clamp_int(x, lo, hi, default=0):
    try:
        x = int(x)
        return max(lo, min(hi, x))
    except:
        return default

def _ensure_list_len(lst, n, fill=""):
    if not isinstance(lst, list):
        lst = []
    lst = [str(x) for x in lst]
    if len(lst) < n:
        lst += [fill] * (n - len(lst))
    return lst[:n]

def analyze_and_draft_tweet(title, summary, link, keyword_hits=None):
    """
    Uses Gemini to:
      1) Score algorithmic-breakthrough relevance for TIG (0-10)
      2) Draft a 1–3 tweet thread if high relevance.
    """
    # Optional cheap signal to help the model calibrate (don’t over-weight).
    keyword_hint = ""
    if keyword_hits is not None:
        keyword_hint = f"\nKEYWORD_HITS (cheap prefilter count): {keyword_hits}\n"

    prompt = f"""
You are an expert in algorithms (theory + systems) and ML optimization.

Use the TIG context below as your rubric.

TIG CONTEXT:
{TIG_CONTEXT}

ARTICLE
TITLE: {title}
SUMMARY: {summary}
LINK: {link}
{keyword_hint}

TASK
Return ONLY valid JSON matching the schema below.

SCORING RUBRIC (0–10)
- 9–10: Clear algorithmic breakthrough + strong evidence (proofs/benchmarks/ablations) + concrete mechanism.
- 7–8: Likely algorithmic improvement but details/evidence partially missing; still meaningful.
- 4–6: Some technical content but mostly application/engineering; novelty unclear.
- 0–3: Hype/product/policy/news; not algorithmic.

REQUIREMENTS
- If relevance_score < 8: set tweet_thread to [] and headline/bullets/tig_link to "".
- If relevance_score >= 8: produce:
  - headline: <= 180 chars
  - bullets: exactly 3 items, each <= 180 chars, in order:
      1) Novelty
      2) Constraint removed
      3) Why it matters economically
  - tig_link: 1–2 sentences, <= 320 chars total
  - tweet_thread: 1–3 tweets; each <= 260 chars; no hashtags; no emoji spam; no hype.
- Evidence: include an 'evidence' field listing up to 3 concrete claims found in the summary/title (numbers, comparisons, guarantees). If none, say ["No concrete evidence in summary"] and penalize score.

OUTPUT JSON SCHEMA
{{
  "relevance_score": <int 0-10>,
  "reasoning": "<1-2 sentences explaining the score>",
  "evidence": ["<claim1>", "<claim2>", "<claim3>"],
  "headline": "<string or empty>",
  "bullets": ["<b1>", "<b2>", "<b3>"] ,
  "tig_link": "<string or empty>",
  "tweet_thread": ["<tweet1>", "<tweet2>", "<tweet3>"]
}}
"""

    try:
        response = genai_client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.4,   # keeps it consistent
            },
        )

        data = _extract_json_object(getattr(response, "text", "") or "")
        if not isinstance(data, dict):
            return {
                "relevance_score": 0,
                "reasoning": "Model did not return valid JSON.",
                "evidence": ["No concrete evidence in summary"],
                "headline": "",
                "bullets": ["", "", ""],
                "tig_link": "",
                "tweet_thread": []
            }

        score = _clamp_int(data.get("relevance_score", 0), 0, 10, default=0)

        # Normalize fields
        reasoning = str(data.get("reasoning", "")).strip()
        evidence = data.get("evidence", ["No concrete evidence in summary"])
        if not isinstance(evidence, list) or len(evidence) == 0:
            evidence = ["No concrete evidence in summary"]
        evidence = [str(x).strip() for x in evidence[:3]]

        if score < 8:
            # Enforce contract: no drafts for low score
            return {
                "relevance_score": score,
                "reasoning": reasoning,
                "evidence": evidence,
                "headline": "",
                "bullets": ["", "", ""],
                "tig_link": "",
                "tweet_thread": []
            }

        headline = str(data.get("headline", "")).strip()[:180]
        bullets = data.get("bullets", ["", "", ""])
        bullets = _ensure_list_len(bullets, 3, fill="")
        bullets = [b.strip()[:180] for b in bullets]

        tig_link = str(data.get("tig_link", "")).strip()[:320]

        tweet_thread = data.get("tweet_thread", [])
        tweet_thread = _ensure_list_len(tweet_thread, 3, fill="")
        # Trim trailing empties to avoid posting blank tweets
        tweet_thread = [t.strip()[:260] for t in tweet_thread if t and t.strip()]

        return {
            "relevance_score": score,
            "reasoning": reasoning,
            "evidence": evidence,
            "headline": headline,
            "bullets": bullets,
            "tig_link": tig_link,
            "tweet_thread": tweet_thread
        }

    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return {
            "relevance_score": 0,
            "reasoning": f"Gemini call failed: {e}",
            "evidence": ["No concrete evidence in summary"],
            "headline": "",
            "bullets": ["", "", ""],
            "tig_link": "",
            "tweet_thread": []
        }

from datetime import datetime, timedelta
from dateutil import parser
import time
import math

# --- HELPERS ---

def parse_pub_date(entry):
    """
    Robustly extract a datetime from RSS entry fields.
    Returns (pub_date_or_None, pub_date_str_for_display)
    """
    for field in ("published", "updated", "created"):
        if hasattr(entry, field):
            raw = getattr(entry, field)
            try:
                dt = parser.parse(raw)
                # normalize tz-aware -> naive local comparable
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt, raw
            except:
                continue
    return None, ""

def recency_score(pub_date, now=None, recent_days=7, max_points=2.0):
    """
    Returns a small bonus in [0, max_points] based on age.
    - Strong preference for last `recent_days`
    - Smooth decay after that (older still eligible)
    """
    if now is None:
        now = datetime.now()
    if pub_date is None:
        return 0.0

    age_days = (now - pub_date).total_seconds() / 86400.0
    if age_days < 0:
        # future timestamp oddity; don't over-reward
        age_days = 0

    # Within the recent window: near-max bonus, tapering slightly
    if age_days <= recent_days:
        # Linear taper from max_points down to ~70% within the week
        return max_points * (1.0 - 0.3 * (age_days / recent_days))

    # Older than window: exponential decay, never negative
    # 14-day half-life after the first week (tweakable)
    half_life = 28.0
    decay = 0.5 ** ((age_days - recent_days) / half_life)
    return max_points * 0.7 * decay

def keyword_score(hits, max_points=1.0, sat_at=6):
    """
    Small bonus for more keyword hits (saturating).
    """
    if hits <= 0:
        return 0.0
    return max_points * min(1.0, hits / float(sat_at))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# --- MAIN ---

import json
import time
from datetime import datetime

def main(
    max_analyses=30,
    algo_threshold=8,
    out_json_path="tig_digest.json",
    out_txt_path="tig_digest.txt",
    write_txt=True,
    verbose=False
):
    now = datetime.now()
    run_meta = {
        "run_at": now.isoformat(timespec="seconds"),
        "feeds_count": len(RSS_FEEDS),
        "max_analyses": max_analyses,
        "algo_threshold": algo_threshold,
    }

    print(f"🕵️ TIG Agent starting... ({run_meta['run_at']})")

    # 1) Collect candidates (keyword-filtered) and pre-rank
    candidates = []
    seen = set()

    feed_errors = 0
    entries_seen = 0

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            feed_errors += 1
            if verbose:
                print(f"[feed error] {feed_url}: {e}")
            continue

        for entry in getattr(feed, "entries", []):
            entries_seen += 1
            title = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "").strip()
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue

            key = link or title.lower()
            if key in seen:
                continue
            seen.add(key)

            text_blob = (title + " " + summary).lower()
            hits = sum(1 for k in KEYWORDS if k in text_blob)
            if hits == 0:
                continue

            pub_date, pub_raw = parse_pub_date(entry)
            rscore = recency_score(pub_date, now=now, recent_days=7, max_points=2.0)
            kscore = keyword_score(hits, max_points=1.0, sat_at=6)
            pre_score = rscore + kscore

            candidates.append({
                "title": title,
                "summary": summary,
                "link": link,
                "pub_date": pub_date.isoformat() if pub_date else None,
                "pub_raw": pub_raw,
                "hits": hits,
                "pre_score": round(pre_score, 3),
                "feed_url": feed_url,
            })

    if not candidates:
        payload = {
            "meta": run_meta | {
                "entries_seen": entries_seen,
                "feed_errors": feed_errors,
                "candidates_after_keywords": 0,
                "analyzed": 0,
                "high_signal": 0,
            },
            "high_signal_items": [],
            "all_analyzed_items": [],
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if write_txt:
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write("No candidates matched keyword prefilter.\n")
        print("Done. No candidates matched keyword prefilter.")
        print(f"Saved: {out_json_path}" + (f" and {out_txt_path}" if write_txt else ""))
        return

    # 2) Sort and limit analyses (cost control)
    candidates.sort(key=lambda x: x["pre_score"], reverse=True)
    candidates = candidates[:max_analyses]

    # 3) Analyze with Gemini and build outputs
    all_analyzed = []
    high_signal = []

    for c in candidates:
        analysis = analyze_and_draft_tweet(
            c["title"],
            c["summary"],
            c["link"],
            keyword_hits=c["hits"]
        )

        if not analysis:
            item = {
                **c,
                "algo_score": 0,
                "recency_bonus": None,
                "keyword_bonus": None,
                "final_score": 0,
                "analysis_failed": True,
                "reasoning": "analysis failed",
                "evidence": [],
                "headline": "",
                "bullets": ["", "", ""],
                "tig_link": "",
                "tweet_thread": [],
            }
            all_analyzed.append(item)
            time.sleep(1)
            continue

        algo_score = float(analysis.get("relevance_score", 0))
        pub_date_dt = datetime.fromisoformat(c["pub_date"]) if c["pub_date"] else None
        rbonus = recency_score(pub_date_dt, now=now, recent_days=7, max_points=2.0)
        kbonus = keyword_score(c["hits"], max_points=1.0, sat_at=6)
        final_score = clamp(algo_score + rbonus + kbonus, 0, 13)

        item = {
            **c,
            "algo_score": int(algo_score),
            "recency_bonus": round(rbonus, 3),
            "keyword_bonus": round(kbonus, 3),
            "final_score": round(final_score, 3),
            "analysis_failed": False,
            "reasoning": str(analysis.get("reasoning", "")).strip(),
            "evidence": analysis.get("evidence", [])[:3],
            "headline": analysis.get("headline", ""),
            "bullets": analysis.get("bullets", ["", "", ""])[:3],
            "tig_link": analysis.get("tig_link", ""),
            "tweet_thread": analysis.get("tweet_thread", [])[:3],
        }

        all_analyzed.append(item)

        if item["algo_score"] >= algo_threshold or (item["algo_score"] >= 6 and item["final_score"] > 9):
            high_signal.append(item)

        time.sleep(1)

    # 4) Sort results by final_score so “best” appears first
    all_analyzed.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    high_signal.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    payload = {
        "meta": run_meta | {
            "entries_seen": entries_seen,
            "feed_errors": feed_errors,
            "candidates_after_keywords": len(candidates),
            "analyzed": len(all_analyzed),
            "high_signal": len(high_signal),
        },
        "high_signal_items": high_signal,
        "all_analyzed_items": all_analyzed,
    }

    # 5) Write JSON summary
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 6) Optional: write a human-readable digest
    if write_txt:
        lines = []
        lines.append(f"TIG Digest — {run_meta['run_at']}")
        lines.append(f"Analyzed: {len(all_analyzed)} | High-signal: {len(high_signal)}")
        lines.append("")

        if not high_signal:
            lines.append("No high-signal items found.")
        else:
            for i, it in enumerate(high_signal, start=1):
                date_str = it["pub_date"][:10] if it["pub_date"] else "Unknown date"
                lines.append(f"{i}. {it['title']} ({date_str})")
                lines.append(f"   final={it['final_score']}/13 algo={it['algo_score']}/10 hits={it['hits']}")
                if it.get("headline"):
                    lines.append(f"   Headline: {it['headline']}")
                bullets = [b for b in it.get("bullets", []) if b]
                for b in bullets:
                    lines.append(f"   - {b}")
                if it.get("tig_link"):
                    lines.append(f"   TIG: {it['tig_link']}")
                if it.get("tweet_thread"):
                    lines.append("   Tweets:")
                    for t in it["tweet_thread"]:
                        lines.append(f"     • {t}")
                lines.append(f"   Link: {it['link']}")
                lines.append("")

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")

    # 7) Minimal terminal output
    print(
        f"Done. Candidates={len(candidates)} Analyzed={len(all_analyzed)} "
        f"High-signal={len(high_signal)} | Saved: {out_json_path}"
        + (f", {out_txt_path}" if write_txt else "")
    )

if __name__ == "__main__":
    main()