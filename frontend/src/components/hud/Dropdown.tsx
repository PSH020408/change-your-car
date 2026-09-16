"use client";
import { useEffect, useId, useRef, useState } from "react";

export interface Option<T extends string | number> { value: T; label: string; hint?: string }

/**
 * A select that always opens DOWNWARD, scrolls inside itself and never leaves
 * the window. The native <select> on macOS anchors the popup on the selected
 * row and spills above the page with long lists (the LAP list has ~90 rows).
 */
export function Dropdown<T extends string | number>({ label, value, options, onChange, disabled, width = 180 }: {
  label: string; value: T; options: Option<T>[]; onChange: (v: T) => void; disabled?: boolean; width?: number;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const id = useId();
  const current = options.find((o) => String(o.value) === String(value));

  useEffect(() => {
    if (!open) return;
    const idx = Math.max(0, options.findIndex((o) => String(o.value) === String(value)));
    setActive(idx);
    const onDoc = (e: MouseEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc); document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onKey); };
  }, [open, options, value]);

  useEffect(() => {
    if (!open) return;
    list.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const pick = (o: Option<T>) => { onChange(o.value); setOpen(false); };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (!open && (e.key === "Enter" || e.key === " " || e.key === "ArrowDown")) { e.preventDefault(); setOpen(true); return; }
    if (!open) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(options.length - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
    else if (e.key === "Enter") { e.preventDefault(); if (options[active]) pick(options[active]); }
  };

  return (
    <div ref={root} className="flex flex-col gap-0.5 relative" style={{ width }}>
      <span className="label">{label}</span>
      <button type="button" disabled={disabled || options.length === 0} onClick={() => setOpen((v) => !v)} onKeyDown={onKeyDown}
        aria-haspopup="listbox" aria-expanded={open} aria-controls={id}
        className={`h-7 w-full bg-hud-inset border rounded px-2 pr-6 text-left text-[12px] font-mono text-hud-text truncate relative disabled:opacity-50 ${open ? "border-series-sim" : "border-hud-line2 hover:border-hud-muted"}`}>
        {current?.label ?? String(value)}
        <svg viewBox="0 0 10 10" width={10} height={10} className="absolute right-2 top-1/2 -translate-y-1/2" fill="none" stroke="#8b93a3" strokeWidth={1.4}>
          <path d="M2 3.5 L5 6.5 L8 3.5" />
        </svg>
      </button>
      {open && (
        <ul ref={list} id={id} role="listbox" tabIndex={-1}
          className="absolute left-0 top-full mt-1 z-50 w-full max-h-[320px] overflow-y-auto bg-hud-panel border border-hud-line2 rounded shadow-xl py-1"
          style={{ minWidth: width, width: "max-content", maxWidth: 360 }}>
          {options.map((o, i) => {
            const sel = String(o.value) === String(value);
            return (
              <li key={String(o.value)} role="option" aria-selected={sel}
                onMouseEnter={() => setActive(i)} onMouseDown={(e) => { e.preventDefault(); pick(o); }}
                className={`px-2 py-1 text-[12px] font-mono cursor-pointer whitespace-nowrap flex justify-between gap-3 ${i === active ? "bg-hud-inset" : ""} ${sel ? "text-series-sim" : "text-hud-soft"}`}>
                <span className="truncate">{o.label}</span>
                {o.hint && <span className="text-hud-dim text-[10px] shrink-0">{o.hint}</span>}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
