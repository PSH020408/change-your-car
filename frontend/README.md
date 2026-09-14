# Frontend — 2D Engineering HUD

Next.js App Router + Tailwind. **No 3D libraries.** Track map and car
schematic are hand-authored SVG; telemetry overlays are Canvas 2D
(Recharts is used only for small static panels, if at all).

```
src/
  app/            routes + layout
  components/
    setup/        2D car schematic, sliders, environment panel
    track/        SVG track map + sector delta overlay
    telemetry/    Canvas overlay charts (real vs simulated)
    hud/          shell, status bar, delta readout, engineer log
  lib/            api client, types, stores
  styles/         globals.css + design tokens
```

Performance budget (Phase 8 gate): initial JS < 300 KB gzip,
telemetry re-render < 16 ms, simulate round-trip p95 < 400 ms.
