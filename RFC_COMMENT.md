I have been exploring a custom-scheduler example for changing total per-step
token budgets under mixed prefill/decode load. It uses the pressure predicate
discussed here, with hysteresis, a minimum residence time, and a rate-limited
budget boost for prefills that stop advancing.

There is a material distinction from P-PAS: my prototype caps **all** scheduled
tokens by narrowing `max_num_scheduled_tokens`; it does not implement a separate
aggregate prefill budget. I have corrected the experiment documentation so its
`ppas` comparator is not presented as a reproduction of P-PAS.

The proposed contribution is an opt-in example under
`examples/features/adaptive_budget/`, using the existing `scheduler_cls`
extension point. It does not change the standard scheduler, default behavior,
KV allocation, or queue order. It is limited to synchronous, single-GPU text
generation. The boost offers a progress opportunity, not a starvation bound.

The accompanying validation includes real allocator lifecycle tests, real-model
generation and prefix-cache checks, and randomized-order comparisons with
several fixed budgets. The experiments show a latency trade-off rather than a
universal performance improvement; full measurements and limitations are in
the attached report. AI assistance was used in development and validation.

Would a documented scheduler example be useful here, or would you prefer this
to remain an out-of-tree experiment while the aggregate-prefill-budget API is
discussed? I would like to coordinate rather than duplicate the work in this RFC.

<!-- Before posting: review the implementation yourself, attach the current
REVISION_REPORT.md and evidence archive, and replace this comment with a public
link to your branch or repository. -->
