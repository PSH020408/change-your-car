import type { Config } from "tailwindcss";

/**
 * Cyber-Engineering design tokens.
 * Slate substrate + a strictly limited neon accent set — neon is *data*,
 * never decoration: every accent colour below carries meaning in the HUD.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        hud: {
          void: "#05070B",     // page ground
          panel: "#0B111A",    // panel fill
          line: "#1B2635",     // hairline borders
          muted: "#5A6B80",    // secondary text
          text: "#C9D6E4",     // primary text
        },
        signal: {
          cyan: "#22D3EE",     // baseline / real telemetry
          amber: "#FBBF24",    // simulated telemetry
          lime: "#84CC16",     // time GAINED  (negative delta)
          rose: "#FB7185",     // time LOST    (positive delta)
          violet: "#A78BFA",   // DRS / special states
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.25), 0 0 24px -6px rgba(34,211,238,0.45)",
      },
    },
  },
  plugins: [],
};
export default config;
