"""
Benchmark TTS — speak_interruptible
==============================================
Measures the performance of the `speak_interruptible` method using a simulated stream
of type `Iterator[CreateChatCompletionStreamResponse]` (llama-cpp-python).

Usage:
    python tts_benchmark.py
    python tts_benchmark.py --runs 5 --delay 0.02
"""

import argparse
import statistics
import time
import types
from dataclasses import dataclass, field
from typing import Iterator, List

from majordome import persona

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_SENTENCE = (
    "Ah, bonjour Alexandre ! Entre, entre ! "
    "Tu sens déjà l'odeur des bonnes choses, j'espère ? "
    "Viens t'asseoir là, près de la lumière. "
    "Qu'est-ce qui te ferait envie ce soir ? "
    "On va faire quelque chose de simple, de parfait."
)

DEFAULT_RUNS = 3
DEFAULT_WARMUP_RUNS = 1          # warmup runs (not counted in results)
DEFAULT_TOKEN_DELAY_S = 0.01     # simulated inter-token delay (seconds)
DEFAULT_CHUNK_SIZE = 3           # chars per simulated token


# ---------------------------------------------------------------------------
# Simulated llama-cpp stream
# ---------------------------------------------------------------------------

def _make_chunk(content: str) -> dict:
    """Build a fake CreateChatCompletionStreamResponse dict."""
    return {
        "choices": [
            {
                "delta": {"content": content},
                "finish_reason": None,
                "index": 0,
            }
        ],
        "id": "benchmark-chunk",
        "model": "benchmark",
        "object": "chat.completion.chunk",
    }


def make_fake_stream(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    delay: float = DEFAULT_TOKEN_DELAY_S,
) -> Iterator[dict]:
    """Yield token-sized chunks that mimic llama-cpp streaming output."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        if delay > 0:
            time.sleep(delay)
        yield _make_chunk(chunk)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    run_id: int
    init_duration_s: float
    warmup_duration_s: float
    speak_duration_s: float
    ttfb_s: float              # Time-To-First-Byte: first audio chunk ready
    full_text: str
    tokens_count: int
    chars_count: int

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_count / self.speak_duration_s if self.speak_duration_s else 0.0

    @property
    def chars_per_second(self) -> float:
        return self.chars_count / self.speak_duration_s if self.speak_duration_s else 0.0


@dataclass
class BenchmarkSuite:
    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, r: BenchmarkResult) -> None:
        self.results.append(r)

    def summary(self) -> dict:
        if not self.results:
            return {}
        durations = [r.speak_duration_s for r in self.results]
        ttfbs = [r.ttfb_s for r in self.results]
        return {
            "runs": len(self.results),
            "speak_duration": {
                "min_s":    round(min(durations), 4),
                "max_s":    round(max(durations), 4),
                "mean_s":   round(statistics.mean(durations), 4),
                "median_s": round(statistics.median(durations), 4),
                "stdev_s":  round(statistics.stdev(durations), 4) if len(durations) > 1 else 0.0,
            },
            "ttfb": {
                "min_s":    round(min(ttfbs), 4),
                "max_s":    round(max(ttfbs), 4),
                "mean_s":   round(statistics.mean(ttfbs), 4),
            },
            "throughput": {
                "mean_tokens_per_s": round(
                    statistics.mean(r.tokens_per_second for r in self.results), 2
                ),
                "mean_chars_per_s": round(
                    statistics.mean(r.chars_per_second for r in self.results), 2
                ),
            },
        }


# ---------------------------------------------------------------------------
# Instrumented wrappers
# ---------------------------------------------------------------------------

def _patch_tts_audio_queue_for_ttfb(original_speak_interruptible):
    """
    Wrap speak_interruptible to intercept the first audio push
    and record Time-To-First-Byte (TTFB).

    Strategy: bind a patched `put` directly onto the Queue *instance* via
    types.MethodType so every thread that holds a reference to the same
    object (including the TTS/playback threads spawned inside
    speak_interruptible) will call the instrumented version.
    """
    import tts

    _first_audio_ts: List[float] = []
    q = tts.tts_audio_queue  # the actual Queue instance
    original_put = q.__class__.put  # unbound method from Queue class

    def _patched_put(self, item, *args, **kwargs):
        if not _first_audio_ts and item is not None:
            _first_audio_ts.append(time.perf_counter())
        return original_put(self, item, *args, **kwargs)

    def wrapped(stream) -> tuple:
        """Returns (full_text, ttfb_s)."""
        _first_audio_ts.clear()
        # Bind the patch onto the instance so all threads see it
        q.put = types.MethodType(_patched_put, q)
        t0 = time.perf_counter()
        full_text = original_speak_interruptible(stream)
        # Restore: delete the instance-level override, falls back to class
        try:
            del q.put
        except AttributeError:
            pass
        ttfb = (_first_audio_ts[0] - t0) if _first_audio_ts else float("nan")
        return full_text, ttfb

    return wrapped


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    token_delay: float = DEFAULT_TOKEN_DELAY_S,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> BenchmarkSuite:
    import tts

    suite = BenchmarkSuite()

    # ---- 1. Model initialisation ----------------------------------------
    if verbose:
        print("=" * 60)
        print("  TTS — Benchmark speak_interruptible")
        print("=" * 60)
        print(f"\n[1/3] Initialisation du modèle …", end=" ", flush=True)

    t_init_start = time.perf_counter()
    persona.preload()
    tts.preload()
    init_duration = time.perf_counter() - t_init_start

    if verbose:
        print(f"OK  ({init_duration:.3f} s)")

    # ---- 2. Warmup -------------------------------------------------------
    if verbose:
        print(f"[2/3] Warmup ({warmup_runs} run(s)) …", end=" ", flush=True)

    t_warmup_start = time.perf_counter()
    for _ in range(warmup_runs):
        stream = make_fake_stream(BENCHMARK_SENTENCE, chunk_size, delay=0)
        tts.speak_interruptible(stream)
    warmup_duration = time.perf_counter() - t_warmup_start

    if verbose:
        print(f"OK  ({warmup_duration:.3f} s)")

    # ---- 3. Benchmark runs -----------------------------------------------
    if verbose:
        print(f"[3/3] Benchmark ({runs} run(s), token_delay={token_delay}s) …\n")

    instrumented = _patch_tts_audio_queue_for_ttfb(tts.speak_interruptible)

    for i in range(1, runs + 1):
        stream = make_fake_stream(BENCHMARK_SENTENCE, chunk_size, delay=token_delay)
        token_count = len(BENCHMARK_SENTENCE) // chunk_size + (
            1 if len(BENCHMARK_SENTENCE) % chunk_size else 0
        )

        t_start = time.perf_counter()
        full_text, ttfb = instrumented(stream)
        speak_duration = time.perf_counter() - t_start

        result = BenchmarkResult(
            run_id=i,
            init_duration_s=init_duration,
            warmup_duration_s=warmup_duration,
            speak_duration_s=speak_duration,
            ttfb_s=ttfb,
            full_text=full_text,
            tokens_count=token_count,
            chars_count=len(BENCHMARK_SENTENCE),
        )
        suite.add(result)

        if verbose:
            print(
                f"  Run {i:>2}/{runs}  |  speak={speak_duration:.3f}s  "
                f"|  TTFB={ttfb:.3f}s  "
                f"|  {result.chars_per_second:.0f} chars/s"
            )

    return suite


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(suite: BenchmarkSuite) -> None:
    s = suite.summary()
    if not s:
        print("No results.")
        return

    r0 = suite.results[0]
    print("\n" + "=" * 60)
    print("  RAPPORT DE BENCHMARK")
    print("=" * 60)
    print(f"  Phrase testée     : {BENCHMARK_SENTENCE[:60]}…")
    print(f"  Longueur          : {len(BENCHMARK_SENTENCE)} caractères")
    print(f"  Runs comptabilisés: {s['runs']}")
    print()
    print("  ── Initialisation ──────────────────────────────────────")
    print(f"  Init model        : {r0.init_duration_s:.3f} s")
    print(f"  Warmup total      : {r0.warmup_duration_s:.3f} s")
    print()
    print("  ── speak_interruptible ─────────────────────────────────")
    d = s["speak_duration"]
    print(f"  Durée min         : {d['min_s']:.4f} s")
    print(f"  Durée max         : {d['max_s']:.4f} s")
    print(f"  Durée moyenne     : {d['mean_s']:.4f} s")
    print(f"  Durée médiane     : {d['median_s']:.4f} s")
    print(f"  Écart-type        : {d['stdev_s']:.4f} s")
    print()
    t = s["ttfb"]
    print("  ── Latence TTFB (premier audio) ────────────────────────")
    print(f"  TTFB min          : {t['min_s']:.4f} s")
    print(f"  TTFB max          : {t['max_s']:.4f} s")
    print(f"  TTFB moyen        : {t['mean_s']:.4f} s")
    print()
    th = s["throughput"]
    print("  ── Débit ───────────────────────────────────────────────")
    print(f"  Tokens/s          : {th['mean_tokens_per_s']:.1f}")
    print(f"  Chars/s           : {th['mean_chars_per_s']:.1f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_output(suite: BenchmarkSuite) -> bool:
    """
    Vérifie que le texte reconstruit par speak_interruptible correspond
    à la phrase de référence (modulo espaces).
    """
    all_ok = True
    for r in suite.results:
        reconstructed = r.full_text.strip()
        expected = BENCHMARK_SENTENCE.strip()
        if reconstructed != expected:
            print(
                f"[WARN] Run {r.run_id}: texte inattendu.\n"
                f"  Attendu : {expected!r}\n"
                f"  Obtenu  : {reconstructed!r}"
            )
            all_ok = False
    if all_ok:
        print(f"\n[OK] Validation texte : tous les runs correspondent à la phrase de référence.")
    return all_ok


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TTS speak_interruptible"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Nombre de runs de benchmark (défaut: {DEFAULT_RUNS})"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_RUNS,
        help=f"Nombre de runs de warmup (défaut: {DEFAULT_WARMUP_RUNS})"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_TOKEN_DELAY_S,
        help=f"Délai inter-token simulé en secondes (défaut: {DEFAULT_TOKEN_DELAY_S})"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Taille des chunks de token simulés (défaut: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Désactiver la validation du texte reconstitué"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suite = run_benchmark(
        runs=args.runs,
        warmup_runs=args.warmup,
        token_delay=args.delay,
        chunk_size=args.chunk_size,
        verbose=True,
    )

    if not args.no_validate:
        validate_output(suite)

    print_report(suite)


if __name__ == "__main__":
    main()