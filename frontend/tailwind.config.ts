import type { Config } from "tailwindcss";

/**
 * Design tokens — Direction A "Console" (design canvas, 2026-09-16).
 * Slate substrate; colour is DATA, never decoration. The chart colours passed
 * the data-viz palette validator on the dark surface (CVD-safe, >= 3:1).
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        hud: {
          void: "#0f1216",     // page ground
          bar: "#12161b",      // top bar
          panel: "#151a20",    // cards
          inset: "#1b2129",    // chips, bars
          line: "#232a33",     // hairlines
          line2: "#2b333e",    // stronger hairline / sliders
          muted: "#8b93a3",    // secondary text
          dim: "#5b6472",      // tertiary text
          text: "#e6e8ec",     // primary text
          soft: "#c3c8d1",     // body text
        },
        series: {
          real: "#c3c2b7",     // the real lap (neutral reference)
          sim: "#3987e5",      // the simulated lap
        },
        status: {
          gain: "#0ca30c",     // time gained (negative delta)
          loss: "#ec835a",     // time lost (positive delta)
          warn: "#e0a54a",
          critical: "#d03b3b",
        },
        grade: {
          a: "#4fc46a", abg: "#0f2a1a", abd: "#1f5a33",
          b: "#6aa6f0", bbg: "#13233a", bbd: "#234a7a",
          c: "#e0a54a", cbg: "#2a2113", cbd: "#5a4620",
        },
      },
      fontFamily: {
        sans: ["var(--font-barlow)", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["var(--font-plex-mono)", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
