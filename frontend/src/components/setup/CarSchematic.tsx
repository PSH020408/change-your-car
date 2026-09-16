import type { CarSetup } from "@/lib/types";

/** Top-view schematic; wings scale with their sliders, ride height with its. */
export function CarSchematic({ setup }: { setup: CarSetup }) {
  const fw = 20 + setup.front_wing * 12;          // element chord
  const rw = 22 + setup.rear_wing * 14;
  const rh = 8 + (1 - setup.ride_height) * 16;     // lower slider = closer to the ground line
  const c = "#5b6472";
  return (
    <svg width="288" height="120" viewBox="0 0 288 120" fill="none" stroke={c} strokeWidth="1.4" aria-label="car schematic">
      <rect x="20" y="52" width="248" height="16" rx="6" stroke="#8b93a3" />
      <rect x={21 - fw / 2} y="30" width={fw} height="60" rx="3" stroke="#3987e5" strokeWidth={1.5 + setup.front_wing} />
      <rect x={266 - rw / 2} y="26" width={rw} height="68" rx="3" stroke="#3987e5" strokeWidth={1.5 + setup.rear_wing * 1.5} />
      <rect x="96" y="40" width="96" height="40" rx="8" />
      <rect x="60" y="20" width="22" height="18" rx="3" /><rect x="60" y="82" width="22" height="18" rx="3" />
      <rect x="206" y="20" width="22" height="18" rx="3" /><rect x="206" y="82" width="22" height="18" rx="3" />
      <path d="M20 112 h248" stroke="#2b333e" strokeDasharray="3 4" />
      <path d={`M144 112 v-${rh}`} stroke={setup.ride_height < 0.25 ? "#d03b3b" : "#e0a54a"} strokeWidth="1.6" />
      <text x="10" y="14" fill="#8b93a3" fontSize="9" fontFamily="var(--font-plex-mono), monospace" stroke="none">FRONT WING</text>
      <text x="222" y="14" fill="#8b93a3" fontSize="9" fontFamily="var(--font-plex-mono), monospace" stroke="none">REAR WING</text>
      <text x="150" y="104" fill={setup.ride_height < 0.25 ? "#d03b3b" : "#e0a54a"} fontSize="9" fontFamily="var(--font-plex-mono), monospace" stroke="none">
        {setup.ride_height < 0.25 ? "bottoming zone" : "ride height"}
      </text>
    </svg>
  );
}
