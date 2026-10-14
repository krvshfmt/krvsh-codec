# krvsh codec documentation

- [design.md](design.md) — rationale, decisions, two-layer architecture, codec math,
  entropy coder, test plan.
- [krvsh-codec-spec.md](krvsh-codec-spec.md) — the krvsh codec: init/frame payload byte
  layout, dtype boundary, PCA + quant + range coder.

The codec-agnostic container format it rides in is a separate project (the `krvs`
container, imported as a dependency); its spec lives in that repo
(`docs/container-spec.md`).
