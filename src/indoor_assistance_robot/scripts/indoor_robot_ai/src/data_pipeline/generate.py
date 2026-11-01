"""Reproducible domain-command generation with provenance and family ids.

Synthetic templates are labeled as such. They are not claimed to equal
naturally spoken human data. Legacy hand-authored examples are imported
with source=legacy_hand_authored.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.lexicon import (
    CatalogEntry,
    location_entries,
    object_entries,
    person_entries,
    primary_surfaces,
)
from src.data_pipeline.schema import DatasetRecord
from src.nlu.normalization import normalize_text

RAW_DIR = PROJECT_ROOT / "data" / "nlu_v2" / "raw"
LEGACY_TRAIN = PROJECT_ROOT / "data" / "nlu" / "train.jsonl"
LEGACY_TEST = PROJECT_ROOT / "data" / "nlu" / "test.jsonl"

# Locations that make sense as RETURN destinations.
_RETURN_LOCATIONS = {"charging_station", "home", "base"}


def _family(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{digest}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _punctuate(text: str, style: str) -> str:
    text = text.strip()
    if style == "question":
        return text if text.endswith("?") else f"{text}?"
    if style == "period":
        return text if text.endswith(".") else f"{text}."
    return text


def _asr_noise(text: str, rng: random.Random) -> str:
    """Light ASR-like corruption; labeled asr_noise_augmented."""
    words = text.split()
    if len(words) < 2:
        return text
    ops = rng.randint(1, 2)
    out = list(words)
    for _ in range(ops):
        i = rng.randrange(len(out))
        op = rng.choice(["drop", "dup", "swap", "typo"])
        if op == "drop" and len(out) > 2:
            out.pop(i)
        elif op == "dup":
            out.insert(i, out[i])
        elif op == "swap" and i + 1 < len(out):
            out[i], out[i + 1] = out[i + 1], out[i]
        else:
            w = out[i]
            if len(w) > 3:
                j = rng.randrange(1, len(w) - 1)
                chars = list(w)
                chars[j] = rng.choice("aeiou")
                out[i] = "".join(chars)
    return " ".join(out)


def _load_legacy() -> List[DatasetRecord]:
    records: List[DatasetRecord] = []
    for path, split_hint in ((LEGACY_TRAIN, "legacy_train"), (LEGACY_TEST, "legacy_test")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            text = str(row["text"]).strip()
            intent = str(row["intent"]).strip()
            entities = {str(k): str(v) for k, v in (row.get("entities") or {}).items() if v}
            records.append(
                DatasetRecord(
                    text=text,
                    intent=intent,
                    entities=entities,
                    source="legacy_hand_authored",
                    family_id=_family("legacy", normalize_text(text)),
                    tags=["legacy", split_hint],
                    notes="Imported from data/nlu without re-labeling.",
                )
            )
    return records


def _navigate_templates(
    loc: CatalogEntry, rng: random.Random
) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    """Return (text, family_key, entities, tags)."""
    place = loc.surface
    eid = loc.entity_id
    ents = {"destination": eid, "action": "navigate"}
    patterns = [
        ("imperative", f"Go to the {place}", "period"),
        ("imperative", f"Take me to the {place}", "period"),
        ("imperative", f"Head to the {place}", "period"),
        ("imperative", f"Drive to the {place}", "period"),
        ("imperative", f"Navigate to the {place}", "period"),
        ("polite", f"Could you take me to the {place}", "question"),
        ("polite", f"Please go to the {place}", "period"),
        ("polite", f"Would you mind escorting me to the {place}", "question"),
        ("indirect", f"I need to get to the {place}", "period"),
        ("indirect", f"I should be in the {place}", "period"),
        ("question", f"Can you take me to the {place}", "question"),
        ("formal", f"Proceed to the {place}", "period"),
        ("casual", f"Let's go to the {place}", "period"),
        ("uncommon", f"Roll over to the {place}", "period"),
        ("uncommon", f"Escort me toward the {place}", "period"),
    ]
    out = []
    for tag, body, style in patterns:
        text = _punctuate(_cap(body), style)
        out.append((text, f"nav|{tag}|{eid}", ents, ["navigate", tag]))
    # Missing destination (incomplete, still NAVIGATE for NLU; validation catches).
    if rng.random() < 0.35:
        out.append(
            (
                _punctuate("Go there", "period"),
                f"nav|deixis|{eid}",
                {"action": "navigate"},
                ["navigate", "missing_slot", "deixis"],
            )
        )
    return out


def _retrieve_templates(
    obj: CatalogEntry,
    src: CatalogEntry,
    dst: Optional[CatalogEntry],
    rng: random.Random,
) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    o, s = obj.surface, src.surface
    ents: Dict[str, str] = {
        "object": obj.entity_id,
        "source_location": src.entity_id,
        "action": "retrieve",
    }
    if dst is not None:
        ents["destination"] = dst.entity_id
    patterns = [
        ("imperative", f"Bring me the {o} from the {s}", "period"),
        ("imperative", f"Fetch the {o} from the {s}", "period"),
        ("imperative", f"Get the {o} from the {s}", "period"),
        ("imperative", f"Grab the {o} from the {s}", "period"),
        ("polite", f"Could you bring the {o} from the {s}", "question"),
        ("polite", f"Please fetch the {o} from the {s}", "period"),
        ("indirect", f"I need the {o} from the {s}", "period"),
        ("question", f"Can you get the {o} from the {s}", "question"),
        ("uncommon", f"Retrieve the {o} from the {s}", "period"),
        ("uncommon", f"Collect the {o} out of the {s}", "period"),
        ("partial", f"Bring me the {o}", "period"),
        ("partial", f"Get the {o}", "period"),
    ]
    out = []
    for tag, body, style in patterns:
        local = dict(ents)
        if tag == "partial":
            local.pop("source_location", None)
            local.pop("destination", None)
        text = _punctuate(_cap(body), style)
        if dst is not None and tag not in {"partial"} and rng.random() < 0.4:
            dest_surf = dst.surface
            text = _punctuate(
                _cap(f"Bring the {o} from the {s} to the {dest_surf}"), "period"
            )
            tag = "full_path"
            local["destination"] = dst.entity_id
        out.append((text, f"ret|{tag}|{obj.entity_id}|{src.entity_id}", local, ["retrieve", tag]))
    return out


def _deliver_templates(
    obj: CatalogEntry, person: CatalogEntry
) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    o, p = obj.surface, person.surface
    ents = {"object": obj.entity_id, "person": person.entity_id, "action": "deliver"}
    patterns = [
        ("imperative", f"Give the {o} to {p}", "period"),
        ("imperative", f"Hand the {o} to {p}", "period"),
        ("imperative", f"Deliver the {o} to {p}", "period"),
        ("polite", f"Please give the {o} to {p}", "period"),
        ("question", f"Can you hand the {o} to {p}", "question"),
        ("indirect", f"{p} needs the {o}", "period"),
        ("uncommon", f"Pass the {o} to {p}", "period"),
    ]
    return [
        (
            _punctuate(_cap(body), style),
            f"del|{tag}|{obj.entity_id}|{person.entity_id}",
            ents,
            ["deliver", tag],
        )
        for tag, body, style in patterns
    ]


def _follow_templates(person: CatalogEntry) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    p = person.surface
    patterns = [
        ("imperative", f"Follow {p}", {"person": person.entity_id, "action": "follow"}, "period"),
        ("imperative", f"Follow me", {"person": "user", "action": "follow"}, "period"),
        ("polite", f"Please follow {p}", {"person": person.entity_id, "action": "follow"}, "period"),
        ("uncommon", f"Shadow {p}", {"person": person.entity_id, "action": "follow"}, "period"),
        ("uncommon", f"Accompany me", {"person": "user", "action": "follow"}, "period"),
        ("question", f"Can you follow me", {"person": "user", "action": "follow"}, "question"),
    ]
    out = []
    for tag, body, ents, style in patterns:
        out.append(
            (
                _punctuate(_cap(body), style),
                f"fol|{tag}|{ents['person']}",
                ents,
                ["follow", tag],
            )
        )
    return out


def _stop_templates() -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    patterns = [
        ("imperative", "Stop", {"action": "stop"}),
        ("imperative", "Stop moving", {"action": "stop"}),
        ("imperative", "Halt", {"action": "stop"}),
        ("imperative", "Freeze", {"action": "stop"}),
        ("imperative", "Abort", {"action": "stop"}),
        ("imperative", "Cancel that", {"action": "stop"}),
        ("negation", "Don't go to the kitchen", {"action": "stop"}),
        ("negation", "Do not take me to the bedroom", {"action": "stop"}),
        ("negation", "Don't move", {"action": "stop"}),
        ("polite", "Please stop", {"action": "stop"}),
        ("urgent", "Stop right now", {"action": "stop"}),
    ]
    return [
        (_punctuate(_cap(body), "period"), f"stop|{tag}", ents, ["stop", tag])
        for tag, body, ents in patterns
    ]


def _return_templates(loc: CatalogEntry) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    place = loc.surface
    ents = {"destination": loc.entity_id, "action": "return"}
    patterns = [
        ("imperative", f"Return to the {place}", "period"),
        ("imperative", f"Go back to the {place}", "period"),
        ("polite", f"Please return to the {place}", "period"),
        ("indirect", f"I want you back at the {place}", "period"),
        ("uncommon", f"Head back to the {place}", "period"),
    ]
    return [
        (
            _punctuate(_cap(body), style),
            f"retn|{tag}|{loc.entity_id}",
            ents,
            ["return", tag],
        )
        for tag, body, style in patterns
    ]


def _unsupported_templates(rng: random.Random) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    """Project-specific negatives — not CLINC label transplants."""
    phrases = [
        ("phys_impossible", "Fly to the kitchen", {}),
        ("phys_impossible", "Do a backflip", {}),
        ("phys_impossible", "Jump over the table", {}),
        ("household", "Cook dinner", {}),
        ("household", "Clean the floor", {}),
        ("household", "Water the plants", {}),
        ("household", "Make me a sandwich from scratch", {}),
        ("manip_unsupported", "Paint the wall", {}),
        ("manip_unsupported", "Unlock the door", {}),
        ("manip_unsupported", "Turn off the air conditioner", {}),
        ("chat", "What is the weather today", {}),
        ("chat", "Tell me a joke", {}),
        ("chat", "How are you feeling", {}),
        ("chat", "Compose a poem about robots", {}),
        ("digital", "Send an email to John", {}),
        ("digital", "Hack the wifi", {}),
        ("digital", "Start a video call", {}),
        ("digital", "Translate this sentence", {}),
        ("digital", "Order a pizza", {}),
        ("digital", "Book a taxi", {}),
        ("object_but_bad_action", "Sing to the water bottle", {}),
        ("object_but_bad_action", "Dance with the red cup", {}),
        ("negated_stop", "Don't stop moving", {}),
        ("contradiction", "Go to the kitchen but stay here", {}),
        ("unrelated", "Who won the football match", {}),
        ("unrelated", "Recite the national anthem", {}),
    ]
    out = []
    for tag, body, ents in phrases:
        style = "question" if body.endswith(("?",)) or body.lower().startswith(("what", "how", "who")) else "period"
        text = _punctuate(_cap(body), style)
        out.append((text, f"oos|{tag}|{normalize_text(body)}", ents, ["unsupported", tag]))
        if rng.random() < 0.3:
            noisy = _asr_noise(text, rng)
            if normalize_text(noisy) != normalize_text(text):
                out.append(
                    (
                        noisy,
                        f"oos|{tag}|noise|{normalize_text(body)}",
                        ents,
                        ["unsupported", tag, "asr_noise"],
                    )
                )
    return out


def _multistep_templates(
    loc_a: CatalogEntry, loc_b: CatalogEntry
) -> List[Tuple[str, str, Dict[str, str], List[str], List[Dict]]]:
    text = _punctuate(
        _cap(f"Go to the {loc_a.surface} and then return to the {loc_b.surface}"),
        "period",
    )
    task_order = [
        {
            "intent": "NAVIGATE",
            "entities": {"destination": loc_a.entity_id, "action": "navigate"},
        },
        {
            "intent": "RETURN",
            "entities": {"destination": loc_b.entity_id, "action": "return"},
        },
    ]
    # Primary intent for single-label classifiers: first actionable step.
    return [
        (
            text,
            f"multi|nav_return|{loc_a.entity_id}|{loc_b.entity_id}",
            {"destination": loc_a.entity_id, "action": "navigate"},
            ["multistep", "navigate"],
            task_order,
        )
    ]


def _human_authored_edge_cases() -> List[DatasetRecord]:
    """Curated edge cases written for this project (not LLM-labeled)."""
    rows = [
        ("Take me to the kitchen.", "NAVIGATE", {"destination": "kitchen", "action": "navigate"}, ["human", "canonical"]),
        ("Don't take me to the kitchen.", "STOP", {"action": "stop"}, ["human", "negation"]),
        ("Stop moving.", "STOP", {"action": "stop"}, ["human", "stop"]),
        ("Don't stop moving.", "UNSUPPORTED", {}, ["human", "negated_stop"]),
        ("Bring me the bottle.", "RETRIEVE", {"object": "bottle", "action": "retrieve"}, ["human", "missing_source"]),
        ("Bring the bottle from the kitchen to the bedroom.", "RETRIEVE", {
            "object": "bottle",
            "source_location": "kitchen",
            "destination": "bedroom",
            "action": "retrieve",
        }, ["human", "full_retrieve"]),
        ("Go to the kitchen and then return to the charging station.", "NAVIGATE", {
            "destination": "kitchen",
            "action": "navigate",
        }, ["human", "multistep"]),
        ("Take it there.", "RETRIEVE", {"action": "retrieve"}, ["human", "coref", "ambiguous"]),
        ("Follow John.", "FOLLOW", {"person": "john", "action": "follow"}, ["human", "follow"]),
        ("Give the medicine to the nurse.", "DELIVER", {
            "object": "medicine",
            "person": "nurse",
            "action": "deliver",
        }, ["human", "deliver"]),
        ("Return to the charging station.", "RETURN", {
            "destination": "charging_station",
            "action": "return",
        }, ["human", "return"]),
        ("What time is it?", "UNSUPPORTED", {}, ["human", "chat"]),
    ]
    records = []
    for text, intent, ents, tags in rows:
        records.append(
            DatasetRecord(
                text=text,
                intent=intent,
                entities=ents,
                source="human_authored",
                family_id=_family("human", normalize_text(text)),
                tags=tags,
                notes="Manually authored for ontology coverage.",
            )
        )
    return records


def _public_aux_oos_stubs() -> List[DatasetRecord]:
    """Small hand-mapped OOS-style negatives inspired by assistant OOS evals.

    These are NOT CLINC150/SNIPS/MASSIVE dumps. They document the methodology
    of using unrelated assistant requests as negatives without importing
    foreign intent taxonomies as robot labels.
    """
    phrases = [
        "play some jazz music",
        "set a timer for five minutes",
        "add milk to my shopping list",
        "what is the capital of france",
        "find a good italian restaurant nearby",
        "remind me to call mom tomorrow",
        "change the thermostat to seventy two",
        "open the garage door",
        "dim the living room lights",
        "how many calories are in an apple",
    ]
    records = []
    for phrase in phrases:
        text = _punctuate(_cap(phrase), "period")
        records.append(
            DatasetRecord(
                text=text,
                intent="UNSUPPORTED",
                entities={},
                source="public_aux_oos",
                family_id=_family("public_oos", normalize_text(phrase)),
                tags=["unsupported", "aux_oos", "methodology_reference"],
                notes=(
                    "Hand-authored assistant-style OOS inspired by public NLU "
                    "OOS evaluation practice. Not a redistribution of CLINC/SNIPS/MASSIVE."
                ),
            )
        )
    return records


def generate_all(seed: int = 42, max_per_family: int = 2) -> List[DatasetRecord]:
    """Build the full unlabeled-as-split corpus with provenance."""
    rng = random.Random(seed)
    records: List[DatasetRecord] = []
    records.extend(_load_legacy())
    records.extend(_human_authored_edge_cases())
    records.extend(_public_aux_oos_stubs())

    primaries = primary_surfaces()
    objects = primaries["object"]
    locations = [e for e in primaries["location"] if e.entity_id not in {"table"}]
    persons = primaries["person"]
    return_locs = [e for e in locations if e.entity_id in _RETURN_LOCATIONS]
    if not return_locs:
        return_locs = [CatalogEntry("charging_station", "charging station")]

    # Sample catalog combinations (not full cartesian product).
    nav_locs = list(locations)
    rng.shuffle(nav_locs)
    for loc in nav_locs:
        for text, fam_key, ents, tags in _navigate_templates(loc, rng):
            records.append(
                DatasetRecord(
                    text=text,
                    intent="NAVIGATE",
                    entities=ents,
                    source="synthetic_template",
                    family_id=_family("tmpl", fam_key),
                    tags=tags,
                )
            )

    # Cap retrieve/deliver combinations so one intent does not dominate.
    obj_sample = list(objects)
    src_sample = list(locations)
    rng.shuffle(obj_sample)
    rng.shuffle(src_sample)
    retrieve_budget = 0
    for obj in obj_sample[:8]:
        for src in src_sample[:3]:
            if retrieve_budget >= 120:
                break
            dst = rng.choice(locations) if rng.random() < 0.5 else None
            if dst is not None and dst.entity_id == src.entity_id:
                dst = None
            templates = _retrieve_templates(obj, src, dst, rng)
            # Keep a diverse subset of templates per (object, source) pair.
            for text, fam_key, ents, tags in templates[:4]:
                records.append(
                    DatasetRecord(
                        text=text,
                        intent="RETRIEVE",
                        entities=ents,
                        source="synthetic_template",
                        family_id=_family("tmpl", fam_key),
                        tags=tags,
                    )
                )
                retrieve_budget += 1

    deliver_budget = 0
    for obj in objects[:8]:
        for person in persons:
            if deliver_budget >= 100:
                break
            templates = _deliver_templates(obj, person)
            for text, fam_key, ents, tags in templates[:3]:
                records.append(
                    DatasetRecord(
                        text=text,
                        intent="DELIVER",
                        entities=ents,
                        source="synthetic_template",
                        family_id=_family("tmpl", fam_key),
                        tags=tags,
                    )
                )
                deliver_budget += 1

    for person in persons:
        for text, fam_key, ents, tags in _follow_templates(person):
            records.append(
                DatasetRecord(
                    text=text,
                    intent="FOLLOW",
                    entities=ents,
                    source="synthetic_template",
                    family_id=_family("tmpl", fam_key),
                    tags=tags,
                )
            )

    for text, fam_key, ents, tags in _stop_templates():
        records.append(
            DatasetRecord(
                text=text,
                intent="STOP",
                entities=ents,
                source="synthetic_template",
                family_id=_family("tmpl", fam_key),
                tags=tags,
            )
        )

    for loc in return_locs:
        for text, fam_key, ents, tags in _return_templates(loc):
            records.append(
                DatasetRecord(
                    text=text,
                    intent="RETURN",
                    entities=ents,
                    source="synthetic_template",
                    family_id=_family("tmpl", fam_key),
                    tags=tags,
                )
            )

    for text, fam_key, ents, tags in _unsupported_templates(rng):
        source = "asr_noise_augmented" if "asr_noise" in tags else "adversarial_edge"
        if "chat" in tags or "unrelated" in tags:
            source = "adversarial_edge"
        records.append(
            DatasetRecord(
                text=text,
                intent="UNSUPPORTED",
                entities=ents,
                source=source,
                family_id=_family("tmpl", fam_key),
                tags=tags,
            )
        )

    multi_pairs = list(zip(locations[:8], return_locs * 8))
    for loc_a, loc_b in multi_pairs:
        for text, fam_key, ents, tags, task_order in _multistep_templates(loc_a, loc_b):
            records.append(
                DatasetRecord(
                    text=text,
                    intent="NAVIGATE",
                    entities=ents,
                    source="synthetic_template",
                    family_id=_family("tmpl", fam_key),
                    tags=tags,
                    task_order=task_order,
                )
            )

    # Controlled paraphrases: synonym swaps within family (still synthetic).
    paraphrase_pool = [r for r in records if r.source == "synthetic_template"]
    rng.shuffle(paraphrase_pool)
    synonym_map = {
        r"\bgo to\b": "head to",
        r"\bbring\b": "fetch",
        r"\bget\b": "grab",
        r"\bgive\b": "hand",
        r"\bplease\b": "kindly",
        r"\bcould you\b": "would you",
    }
    added = 0
    for base in paraphrase_pool:
        if added >= 400:
            break
        text = base.text
        key = rng.choice(list(synonym_map.keys()))
        if not re.search(key, text, flags=re.I):
            continue
        new_text = re.sub(key, synonym_map[key], text, count=1, flags=re.I)
        if normalize_text(new_text) == normalize_text(text):
            continue
        records.append(
            DatasetRecord(
                text=_cap(new_text) if new_text[:1].islower() else new_text,
                intent=base.intent,
                entities=dict(base.entities),
                source="synthetic_paraphrase",
                family_id=base.family_id,  # same family → same split
                tags=list(base.tags) + ["paraphrase"],
                task_order=base.task_order,
                notes="Rule-based synonym paraphrase of a synthetic template.",
            )
        )
        added += 1

    # ASR noise on a subset of supported commands.
    supported = [r for r in records if r.intent != "UNSUPPORTED" and r.source.startswith("synthetic")]
    rng.shuffle(supported)
    for base in supported[:250]:
        noisy = _asr_noise(base.text, rng)
        if normalize_text(noisy) == normalize_text(base.text):
            continue
        records.append(
            DatasetRecord(
                text=noisy,
                intent=base.intent,
                entities=dict(base.entities),
                source="asr_noise_augmented",
                family_id=base.family_id,
                tags=list(base.tags) + ["asr_noise"],
                notes="Light character/word noise approximating ASR errors.",
            )
        )

    # Deduplicate exact normalized text preferring human/legacy over synthetic.
    priority = {
        "human_authored": 0,
        "legacy_hand_authored": 1,
        "public_aux_oos": 2,
        "adversarial_edge": 3,
        "synthetic_template": 4,
        "synthetic_paraphrase": 5,
        "asr_noise_augmented": 6,
    }
    best: Dict[str, DatasetRecord] = {}
    for rec in records:
        key = normalize_text(rec.text)
        if not key:
            continue
        prev = best.get(key)
        if prev is None or priority.get(rec.source, 9) < priority.get(prev.source, 9):
            best[key] = rec

    deduped = list(best.values())
    # Cap near-duplicates per family for train diversity without trivial copies.
    by_family: Dict[str, List[DatasetRecord]] = {}
    for rec in deduped:
        by_family.setdefault(rec.family_id, []).append(rec)
    capped: List[DatasetRecord] = []
    for fam, group in by_family.items():
        group.sort(key=lambda r: (priority.get(r.source, 9), r.text))
        # Keep all human/legacy/adversarial; cap heavy synthetic families.
        keep = []
        synth = []
        for r in group:
            if r.source in {
                "human_authored",
                "legacy_hand_authored",
                "public_aux_oos",
                "adversarial_edge",
            }:
                keep.append(r)
            else:
                synth.append(r)
        keep.extend(synth[: max(max_per_family, 1)])
        capped.extend(keep)

    capped.sort(key=lambda r: (r.intent, r.source, r.text))
    return capped


def write_raw_corpus(records: Sequence[DatasetRecord], path: Optional[Path] = None) -> Path:
    out = path or (RAW_DIR / "corpus.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out


def load_raw_corpus(path: Optional[Path] = None) -> List[DatasetRecord]:
    src = path or (RAW_DIR / "corpus.jsonl")
    records: List[DatasetRecord] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                DatasetRecord(
                    text=row["text"],
                    intent=row["intent"],
                    entities=row.get("entities") or {},
                    source=row.get("source", "synthetic_template"),
                    family_id=row.get("family_id", ""),
                    tags=list(row.get("tags") or []),
                    task_order=row.get("task_order"),
                    notes=row.get("notes", ""),
                    split=row.get("split", ""),
                )
            )
    return records
