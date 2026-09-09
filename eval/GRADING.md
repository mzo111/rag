# Relevance grading rubric

Written after a blind re-grade of 40 judgments measured Cohen's κ = 0.40 on raw
grades and κ = 0.50 on the relevant/not-relevant collapse. Grade 0 reproduced at
90%; grade 1 reproduced at 45% and scattered across all three values. The
instability was in the definition of grade 1, not in any single borderline call,
so this document defines it.

Judgments made before this rubric existed are kept unchanged in git history. The
comparison between the pre-rubric and post-rubric judgment sets is a result, not
a cleanup.

## The question being answered

For each candidate page, against the query as literally written:

> If this were the only page a user got back, how much closer are they to their
> answer?

Not "is this page about the same topic." Not "would a reasonable person also
want this." Only how far it moves the user toward the specific thing asked.

## Grades

**2 — the page carries the answer.**
The specific thing the query asks for is present on this page, and a user
reading it stops looking. The page's primary subject is the thing asked about,
or the answer sits plainly in its body. More than one page can be a 2 when the
documentation genuinely answers the question in more than one place.

**1 — the page carries a pointer to the answer, or one necessary piece of it.**
The user does not finish here, but they leave with something load-bearing: the
name of the function they actually need, an index entry that lists it, a
prerequisite concept the answer depends on, or one component of a multi-part
question. The test is that a specific next step comes out of reading it.

**0 — the page does not move the user.**
Same library, same module, same vocabulary, and none of it helps. A user who
read only this page would restart their search from where they began.

## The 1 / 0 boundary

This is the boundary that failed. The operational rule:

> A page is a 1 only if it contains the target — the symbol, the concept, or a
> direct pointer to it. A page that discusses the same area without containing
> the target is a 0.

Shared vocabulary is not containment. A page that mentions `torch.nn.functional`
while explaining something else does not contain `cross_entropy`'s
`label_smoothing` argument, and does not point at it.

The practical test: **name the next step.** If you can say "they'd read this,
then go to X," it is a 1 and X is the thing that makes it so. If the honest
sentence is "they'd read this, then go back to the search box," it is a 0.

## The 1 / 2 boundary

> A page is a 2 when the query's specific ask is answered on the page. It is a 1
> when the page is about the right area but the specific ask is answered
> elsewhere.

For a query naming an argument, a parameter, or a return value, a 2 requires
that argument, parameter, or return value to be documented on the page. A page
about the right function that does not cover the asked-about detail is a 1.

For conceptual and tutorial queries, a 2 requires the page to explain the
mechanism asked about, not merely use it.

## Tie-break

**When torn between two grades, take the lower one.**

Applied every time, this is a stated convention and the metrics stay
interpretable. Applied selectively, it is noise. If a call takes more than about
fifteen seconds, that is the definition of torn — go lower and move on.

## Worked examples

All drawn from the existing judgment set.

**Query (api_lookup):** `torch.nn.functional.cross_entropy label_smoothing argument`

| Page | Grade | Why |
|---|---|---|
| `torch.nn.functional.cross_entropy` | 2 | Named function, and `label_smoothing` is documented in its parameter list. |
| `torch.nn.functional` → Loss functions | 1 | Index page listing `cross_entropy`. Contains a direct pointer; the user's next step is named. |
| `torch.nn.functional.linear_cross_entropy` | 1 | Different function, but it takes `label_smoothing` and sits adjacent in the API. Contains the concept, not the asked-about page. |
| "What is torch.nn really?" | 0 | Beginner walkthrough mentioning `torch.nn.functional` in passing. Does not contain the argument or point at it. |
| Tensor Parallel tutorial → Apply Loss Parallel | 0 | About sharding loss across devices. Shares the word "loss" and nothing else. |
| `torch.nn.functional.gelu` | 0 | Same module, unrelated function. |

The two 0s in that table were graded 1 in round one. Under this rubric they are
0s, and that change is the rubric doing its job.

**Query (api_lookup):** `what does torch.topk return`

| Page | Grade | Why |
|---|---|---|
| `torch.topk` | 2 | Return value documented on the page. |
| `torch.Tensor.topk` | 2 | Method form of the same call, independently answers it. |
| `torch` → Comparison Ops | 2 | Section documenting the return signature directly. |
| `torch.Tensor` class reference | 1 | Index listing `topk` among methods. Pointer, not answer. |
| `SparseSemiStructuredTensorCUTLASS` | 0 | Keyword collision on "tensor". |

## Application notes

- Grade against the query **as written**, not the question you think was meant.
- Ignore `found_by`. Which retriever surfaced a page is not evidence about it.
- Do not balance grades to a quota. Some queries have eight relevant pages and
  some have two.
- A page you never grade counts as 0 in the metrics, so a skip is an assertion.
  Grade every candidate.

## Decisions in this rubric you should confirm or change

Three calls here were mine to draft and yours to own. Read them before applying
this 159 times.

1. **Index and parent pages are 1s, not 0s.** A module index listing the target
   function is treated as containing a pointer. The alternative — that only
   pages with the answer itself count — is defensible and would push a
   substantial share of your existing 1s to 0.

2. **Sibling functions sharing the asked-about argument are 1s.**
   `linear_cross_entropy` gets a 1 for having `label_smoothing`. If you think a
   user asking about one function gains nothing from a different one, that is a
   0 and the rubric should say so.

3. **Multiple 2s per query are allowed.** This inflates recall ceilings —
   your round-one set averaged 7.35 relevant pages per query, which capped R@5
   at 0.710. Restricting 2 to a single best page per query would produce
   cleaner metrics but a less honest description of the documentation.

Whichever way you settle these, the rubric has to say so in one sentence, and it
has to be committed before the re-grade runs.
