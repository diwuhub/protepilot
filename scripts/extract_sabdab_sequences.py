"""
extract_sabdab_sequences.py — Extract antibody sequences from RCSB PDB
======================================================================
Uses SAbDab summary TSV metadata + RCSB PDB FASTA API to build a
training-ready dataset of antibody/nanobody sequences.

Strategy:
  1. Read SAbDab summary → identify PDB codes + chain letters
  2. Batch-fetch FASTA sequences from RCSB PDB API
  3. Match chain letters → extract HC/LC sequences
  4. Classify: no-LC → single_domain, scFv → single_domain, HC+LC → canonical_mab
  5. De-duplicate by sequence identity
  6. Output: sabdab_sequences.csv (harmonizer-compatible)

Usage:
    python scripts/extract_sabdab_sequences.py [--max-pdbs N] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ── SSL context: fix macOS Python certificate issue ──────────────
def _make_ssl_context() -> ssl.SSLContext:
    """Create an SSL context that works on macOS.

    Tries certifi first (pip install certifi), then falls back to
    an unverified context with a warning.
    """
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass

    # Try default context — works if system certs are properly configured
    ctx = ssl.create_default_context()
    try:
        import urllib.request
        urllib.request.urlopen(
            Request("https://www.rcsb.org", headers={"User-Agent": "test"}),
            timeout=5, context=ctx,
        )
        return ctx
    except Exception:
        pass

    # Last resort: unverified context (still encrypted, just no cert check)
    log_ssl = logging.getLogger(__name__)
    log_ssl.warning(
        "SSL certificate verification failed. Using unverified HTTPS. "
        "To fix properly: pip3 install certifi"
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


SSL_CTX: Optional[ssl.SSLContext] = None  # initialized lazily

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger("SAbDab.Extractor")

# ── Configuration ──────────────────────────────────────────────────────
RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}/download"
# Alternative URLs to try if primary fails
RCSB_FASTA_ALT_URLS = [
    "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1",  # REST API
]
REQUEST_DELAY = 0.25          # seconds between requests (be polite)
REQUEST_TIMEOUT = 15          # seconds
MAX_RETRIES = 2
USER_AGENT = "ProtePilot/1.0 (ProtePilot; antibody research; contact: academic)"
MIN_SEQ_LENGTH = 50           # ignore very short fragments
MAX_CANONICAL_MAB = 2000      # cap canonical_mab to avoid worsening imbalance

# Amino acid alphabet (filter non-standard residues from PDB)
AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_sequence(seq: str) -> str:
    """Remove non-standard residues and whitespace."""
    return "".join(c for c in seq.upper().strip() if c in AA_VALID)


_error_samples_logged = 0  # track how many detailed errors we've shown


def _fetch_fasta(pdb_id: str) -> Optional[str]:
    """Fetch FASTA text for a PDB entry from RCSB.

    Includes User-Agent header (required by RCSB since ~2024) and
    logs the first few errors with full detail for debugging.
    """
    global _error_samples_logged, SSL_CTX
    if SSL_CTX is None:
        SSL_CTX = _make_ssl_context()
    url = RCSB_FASTA_URL.format(pdb_id=pdb_id.upper())
    headers = {
        "Accept": "text/plain",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CTX) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.code == 404:
                log.debug("PDB %s not found (404)", pdb_id)
                return None
            if _error_samples_logged < 5:
                log.error(
                    "HTTP %d for %s (attempt %d/%d) url=%s resp=%s",
                    e.code, pdb_id, attempt + 1, MAX_RETRIES + 1, url,
                    e.read()[:200] if hasattr(e, "read") else "N/A",
                )
                _error_samples_logged += 1
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            log.warning("HTTP error for %s: %s", pdb_id, e)
            return None
        except (URLError, OSError) as e:
            if _error_samples_logged < 5:
                log.error(
                    "Network error for %s (attempt %d/%d): %s",
                    pdb_id, attempt + 1, MAX_RETRIES + 1, e,
                )
                _error_samples_logged += 1
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            log.warning("Network error for %s: %s", pdb_id, e)
            return None
    return None


def _parse_fasta(fasta_text: str) -> Dict[str, str]:
    """Parse FASTA text → {chain_id: sequence}.

    RCSB FASTA headers look like:
    >9NJ3_1|Chain A|...|Homo sapiens (9606)
    The chain letter is in the "Chain X" part.
    We also handle: >PDB_entity|Chains A,B|...
    """
    chains = {}
    current_id = None
    current_seq = []

    for line in fasta_text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            # Save previous
            if current_id and current_seq:
                seq = _clean_sequence("".join(current_seq))
                if len(seq) >= MIN_SEQ_LENGTH:
                    chains[current_id] = seq
            current_seq = []

            # Parse header for chain letter(s)
            # Format: >PDBID_entity|Chain(s) X[,Y]|description|organism
            parts = line[1:].split("|")
            chain_part = parts[1] if len(parts) > 1 else ""
            # Extract chain letters: "Chain A" → "A", "Chains A,B" → "A"
            chain_letters = []
            for token in chain_part.replace(",", " ").split():
                if len(token) == 1 and token.isalpha():
                    chain_letters.append(token)
            # Use first chain letter as ID, or fall back to entity ID
            if chain_letters:
                current_id = chain_letters[0]
            else:
                # Fall back to entity part from PDBID_entity
                current_id = parts[0].split("_")[-1] if "_" in parts[0] else parts[0]
        else:
            current_seq.append(line)

    # Last entry
    if current_id and current_seq:
        seq = _clean_sequence("".join(current_seq))
        if len(seq) >= MIN_SEQ_LENGTH:
            chains[current_id] = seq

    return chains


def _seq_hash(seq: str) -> str:
    """Short hash for de-duplication."""
    return hashlib.md5(seq.encode()).hexdigest()[:12]


def load_sabdab_metadata(tsv_path: str) -> List[Dict[str, Any]]:
    """Load SAbDab summary TSV and return per-PDB metadata."""
    import csv

    entries = {}  # pdb → first entry
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pdb = row.get("pdb", "").strip().lower()
            if not pdb or pdb in entries:
                continue
            entries[pdb] = {
                "pdb": pdb,
                "hchain": row.get("Hchain", "").strip(),
                "lchain": row.get("Lchain", "").strip(),
                "scfv": row.get("scfv", "").strip().lower() == "true",
                "heavy_subclass": row.get("heavy_subclass", ""),
                "light_ctype": row.get("light_ctype", ""),
                "heavy_species": row.get("heavy_species", ""),
                "antigen_name": row.get("antigen_name", ""),
                "compound": row.get("compound", ""),
            }
    return list(entries.values())


def classify_entry(
    meta: Dict[str, Any], chains: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Classify a SAbDab entry into our molecule_class taxonomy.

    Returns a training-ready dict or None if not useful.
    """
    hchain_letter = meta["hchain"]
    lchain_letter = meta["lchain"]
    is_scfv = meta["scfv"]

    hc_seq = chains.get(hchain_letter, "")
    lc_seq = chains.get(lchain_letter, "") if lchain_letter and lchain_letter != "NA" else ""

    # Must have at least heavy chain
    if not hc_seq:
        return None

    name = f"SAbDab_{meta['pdb'].upper()}"
    species = meta.get("heavy_species", "")

    # Classification logic
    if not lc_seq and not is_scfv:
        # No light chain → single_domain (nanobody/VHH)
        if len(hc_seq) > 250:
            return None  # Too long for single-domain, probably missing chain
        molecule_class = "single_domain"
        hc_sequence = hc_seq
        lc_sequence = ""
    elif is_scfv:
        # scFv → single_domain (single-chain variable fragment)
        molecule_class = "single_domain"
        hc_sequence = hc_seq
        lc_sequence = lc_seq if lc_seq else ""
    elif hc_seq and lc_seq:
        # Standard HC+LC → canonical_mab
        molecule_class = "canonical_mab"
        hc_sequence = hc_seq
        lc_sequence = lc_seq
    else:
        return None

    return {
        "name": name,
        "molecule_class": molecule_class,
        "hc_sequence": hc_sequence,
        "lc_sequence": lc_sequence,
        "source": "sabdab",
        "species": species,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract SAbDab sequences from PDB")
    parser.add_argument(
        "--tsv",
        default="data/external/raw/sabdab_summary_all.tsv",
        help="Path to SAbDab summary TSV",
    )
    parser.add_argument(
        "--output",
        default="data/external/raw/sabdab_sequences.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-pdbs",
        type=int,
        default=0,
        help="Max PDB entries to fetch (0=all)",
    )
    parser.add_argument(
        "--priority",
        choices=["nanobody", "all"],
        default="nanobody",
        help="'nanobody' = only no-LC + scFv entries; 'all' = include standard mAbs",
    )
    args = parser.parse_args()

    # ── Quick connectivity check ──────────────────────────────────────
    log.info("Testing RCSB connectivity with PDB 1IGT ...")
    test_fasta = _fetch_fasta("1IGT")
    if test_fasta:
        log.info("RCSB connectivity OK (%d bytes). Proceeding.", len(test_fasta))
    else:
        log.error(
            "RCSB connectivity FAILED for test PDB 1IGT.\n"
            "  Possible causes:\n"
            "  1. No internet / firewall blocking rcsb.org\n"
            "  2. Proxy stripping User-Agent headers\n"
            "  3. RCSB API is temporarily down\n"
            "  Try: curl -v 'https://www.rcsb.org/fasta/entry/1IGT/download'\n"
            "  If curl also fails, it's a network issue on your side."
        )
        sys.exit(1)

    # Load metadata
    log.info("Loading SAbDab metadata from %s", args.tsv)
    all_entries = load_sabdab_metadata(args.tsv)
    log.info("Loaded %d unique PDB entries", len(all_entries))

    # Filter by priority
    if args.priority == "nanobody":
        entries = [
            e for e in all_entries
            if not e["lchain"] or e["lchain"] == "NA" or e["scfv"]
        ]
        log.info("Filtered to %d nanobody/scFv entries", len(entries))
    else:
        entries = all_entries

    if args.max_pdbs > 0:
        entries = entries[: args.max_pdbs]
        log.info("Capped to %d entries", len(entries))

    # Fetch and classify
    results = []
    seen_hashes: Set[str] = set()
    canonical_count = 0
    fetch_ok = 0
    fetch_fail = 0

    for i, meta in enumerate(entries):
        if i > 0 and i % 100 == 0:
            log.info(
                "Progress: %d/%d fetched (%d ok, %d fail, %d results)",
                i, len(entries), fetch_ok, fetch_fail, len(results),
            )

        fasta = _fetch_fasta(meta["pdb"])
        if not fasta:
            fetch_fail += 1
            continue
        fetch_ok += 1

        chains = _parse_fasta(fasta)
        if not chains:
            continue

        record = classify_entry(meta, chains)
        if not record:
            continue

        # Cap canonical_mab to avoid worsening imbalance
        if record["molecule_class"] == "canonical_mab":
            canonical_count += 1
            if canonical_count > MAX_CANONICAL_MAB:
                continue

        # De-duplicate by HC sequence hash
        h = _seq_hash(record["hc_sequence"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        results.append(record)
        time.sleep(REQUEST_DELAY)

    log.info(
        "Extraction complete: %d results from %d fetched PDBs",
        len(results), fetch_ok,
    )

    # Class distribution
    from collections import Counter
    dist = Counter(r["molecule_class"] for r in results)
    log.info("Class distribution: %s", dict(dist))

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = ["name", "molecule_class", "hc_sequence", "lc_sequence", "source", "species"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log.info("Wrote %s (%d rows)", args.output, len(results))
    return results


if __name__ == "__main__":
    main()
