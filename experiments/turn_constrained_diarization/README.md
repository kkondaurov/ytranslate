# Speaker Identity Linker experiment

This regression harness tests whether episode-wide acoustic identity evidence
repairs systematic speaker attribution failures in ytranslate's local-speaker
mapping and voice-reconciliation pipeline. The reviewed SIL `low` candidate is
now the production implementation for recognized All-In episodes; this folder
retains the frozen comparisons, audit windows, holdouts, and reviewer used to
gate future changes.

## Compared variants

- Current ytranslate attribution with GPT-5.6 Luna `low`
- SIL with GPT-5.6 Luna `low` boundary evidence
- SIL with GPT-5.6 Luna `medium` boundary evidence

The primary comparison is current `low` against SIL `low`. SIL `medium` remains
as a boundary-reasoning sensitivity check. The `none` variant was dropped after
it underperformed both reasoning variants on the Brad audit windows.

## Architecture

1. Luna labels each raw ASR segment boundary as `same`, `uncertain`, or
   `change`. It cannot name speakers.
2. Anonymous turns are split into acoustic units no longer than six seconds and
   embedded locally with Resemblyzer.
3. Trusted references attach names to acoustic identities. Each recurring
   All-In host and audited guest has three clips drawn from reviewed passages
   across episodes, avoiding dependence on one potentially poisoned clip. New
   participants are enrolled only when they are active in the episode baseline;
   an independently transcribed panel can instead use source-backed windows.
4. An explicit `Unknown/External` state competes with the enrolled roster. It is
   emitted only for a sustained run of at least two units and 12 seconds whose
   duration-weighted known-speaker similarity remains below the run threshold.
5. A Viterbi decoder combines acoustic similarity with weak Luna boundary
   priors. Luna can influence continuity, but cannot directly choose a name.
6. Known-speaker pairs become eligible only after at least two strong units and
   20 seconds of episode-wide evidence. Application is narrower: a correction
   must belong to a sustained local-track run with no gap above 60 seconds, or
   independently clear a higher per-unit margin. A weaker short-turn exception
   applies only when the entire candidate run is no longer than three seconds.
7. Units that do not pass an application gate are a strict no-op: every
   overlapped segment keeps its own baseline identity. This prevents scattered
   evidence from different chunk mappings from accumulating into a long false
   rewrite while retaining bounded rapid-exchange corrections.
8. Passage-level windows pass at 90% duration accuracy, reflecting the accepted
   residual risk from brief crosstalk while retaining the exact accuracy and
   per-speaker seconds in the report.
9. Results are scored against the fully audited Gavin episode, 21 manually
   audited Brad windows, four regular-four regression windows, and 23 reviewed
   windows from the first 45 minutes of the core-four episode. Archive holdouts
   add All-In episodes 278 and 279 plus a four-person Uncapped panel. The latter
   two add independent external-voice and speaker-window checks.

The material after 45 minutes in the core-four episode, the three-speaker Big
Technology interview, and the unscored regions of all three archive episodes
remain listening-based holdouts.

## Run

From the repository root:

```bash
.venv/bin/pip install -r experiments/turn_constrained_diarization/requirements.txt
.venv/bin/python experiments/turn_constrained_diarization/turn_constrained_eval.py
```

Every Luna boundary batch is cached under `output/episodes/<episode>/`, so a
partial run can resume. `--force-sil` recomputes SIL while preserving the cached
control and boundary batches.

Start the range-capable reviewer server and open the page:

```bash
.venv/bin/python experiments/turn_constrained_diarization/reviewer_server.py
```

`http://127.0.0.1:8877/experiments/turn_constrained_diarization/reviewer/`
