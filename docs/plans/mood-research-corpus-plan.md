# ExecPlan: Mood Context Corpus

Reference template: `/Users/michaelwall/sfm-chatgpt/PLANS.md` (found but empty, so this plan uses the required `exec-plan` section contract directly).

## Objective
Build a researched mood context corpus for the current mood set. For each mood, provide:
- 25 life situation examples
- 25 human nature examples
- 25 historical event examples
- 25 natural occurrence examples

This yields 100 examples per mood.

## Scope
In scope:
- 10 moods: Ambient, Beautiful, Energetic, Hopeful, Humorous, Intense, Mysterious, Nostalgia, Rhythmic, Sad
- Web research to ground examples in real-world contexts
- One markdown deliverable per mood plus one source index

Out of scope:
- Audio production guidance
- Any references to music, art, or dance
- Mood normalization beyond existing canonical mood names

## Definitions
- Life situations: everyday lived contexts people commonly encounter (family transitions, health changes, work tension, belonging).
- Human nature examples: recurring behavioral or psychological patterns (attachment, fairness, risk-taking, status-seeking, meaning-making).
- Historical event examples: dated human events from collective history.
- Natural occurrence examples: non-human environmental, astronomical, geological, or weather processes.

## Constraints
- Exactly 25 examples per category per mood.
- No references to music, art, or dance.
- Keep language plain and direct.
- Provide citation links for research sources.

## Deliverables
- `/Users/michaelwall/compose-creative-researcher/mood_examples/ambient.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/beautiful.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/energetic.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/hopeful.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/humorous.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/intense.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/mysterious.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/nostalgia.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/rhythmic.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/sad.md`
- `/Users/michaelwall/compose-creative-researcher/mood_examples/SOURCES.md`

## Implementation Plan
1. Gather authoritative web sources for emotional definitions, historical events, and natural processes.
2. Draft mood-specific examples for each category with strict category counts.
3. Write one markdown file per mood with consistent structure.
4. Build a source index mapping source domains to usage.
5. Run validation checks for counts and banned terms.
6. Update this plan with outcomes and retrospective notes.

## Commands And Validation
Expected local checks:
    mkdir -p mood_examples
    rg -n "^- " /Users/michaelwall/compose-creative-researcher/mood_examples/*.md
    rg -n -i "\\b(music|art|dance)\\b" /Users/michaelwall/compose-creative-researcher/mood_examples/*.md

Validation method:
- Each mood file contains four sections and each section has exactly 25 bullet examples.
- No banned terms appear as whole words.
- Source index exists and contains links used for research.

## Acceptance Criteria
- All 10 mood files exist.
- Each mood file includes 100 examples (25 in each category).
- Content remains focused on life, human nature, history, and natural occurrences.
- `SOURCES.md` includes the research links used.

## Risks And Mitigations
- Risk: category leakage (example reads like wrong category).
  Mitigation: strict section headers and category-by-category drafting pass.
- Risk: repetitive phrasing across moods.
  Mitigation: vary scenario framing and event selection by mood.
- Risk: accidental banned terms.
  Mitigation: regex scan before completion.

## Progress
- [x] Plan scaffold created.
- [x] Web research completed.
- [x] All mood files drafted.
- [x] Source index compiled.
- [x] Validation passed.
- [x] Final review completed.

## Surprises & Discoveries
- `PLANS.md` was not present in this workspace root.
- A nearby template path exists but the file is empty.
- A generated-content approach was necessary due corpus scale (1,100 examples).

## Decision Log
- 2026-02-05: Use a separate `mood_examples/` directory to keep the original `moods/` definitions clean.
- 2026-02-05: Canonicalized mood taxonomy to a single `Nostalgia` label (removed `Nostalgic` duplicate).

## Outcomes & Retrospective
- Created 10 mood corpus files in `/Users/michaelwall/compose-creative-researcher/mood_examples/`.
- Created `/Users/michaelwall/compose-creative-researcher/mood_examples/SOURCES.md` with web references.
- Validation confirmed each file has exactly 25 items in each of four categories.
- Validation confirmed no whole-word references to music, art, or dance.
