"use client";
import { create } from "zustand";
import type { BaselineRef, CarSetup, Environment } from "./types";
import { DEFAULT_ENV, DEFAULT_SETUP } from "./types";

interface HudState {
  ref: BaselineRef;
  setup: CarSetup;
  env: Environment;
  setRef: (patch: Partial<BaselineRef>) => void;
  setSetup: (patch: Partial<CarSetup>) => void;
  setEnv: (patch: Partial<Environment>) => void;
  reset: () => void;
}

export const useHud = create<HudState>((set) => ({
  ref: { season: 2024, event: "bahrain_grand_prix", session: "Q", driver: "VER", lap: "representative" },
  setup: DEFAULT_SETUP,
  env: DEFAULT_ENV,
  setRef: (patch) => set((s) => ({ ref: { ...s.ref, ...patch }, setup: DEFAULT_SETUP, env: DEFAULT_ENV })),
  setSetup: (patch) => set((s) => ({ setup: { ...s.setup, ...patch } })),
  setEnv: (patch) => set((s) => ({ env: { ...s.env, ...patch } })),
  reset: () => set({ setup: DEFAULT_SETUP, env: DEFAULT_ENV }),
}));
