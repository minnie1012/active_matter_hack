# active_matter_hack
tumor cell modeling for ucsd active matter hackathon

## Featured videos

- [TME full](outputs/video_for_presentation/tme_full.mp4) — full-stack demo with every mechanism on: EMT, ECM fiber alignment + integrin biphasic traction, hypoxia / HIF response, angiogenesis, and the NK / DC / MDSC immune populations layered on top of the CD8 + tumor base.
- [TME baseline](outputs/video_for_presentation/tme_baseline.mp4) — biophysical control with EMT, fiber/integrin, and hypoxia all switched OFF (same seed, same ECM and immune setup), so any structure in the other runs can be attributed to those three mechanisms.
- [TME only EMT](outputs/videos/tme_only_emt.mp4) — only the EMT switch is on. Tumor cells can transition between epithelial and mesenchymal states (different motility, secretion, and daughter drift), while fiber/integrin coupling and hypoxia signaling stay off.
- [TME with hypoxia](outputs/videos/tme_with_hypoxia.mp4) — full HIF response engaged: low O₂ slows tumor division, reduces CD8 / NK kill probability, makes hypoxic tumor cells secrete VEGF, and lets vessels drift up the VEGF gradient and sprout. Paired with `tme_no_hypoxia.mp4` as the "same plumbing, no signaling" control.
