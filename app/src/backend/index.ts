import { isTauri } from "@tauri-apps/api/core";
import type { Backend } from "./types";

export type { Backend } from "./types";

export function runningInTauri(): boolean {
  try {
    if (isTauri()) return true;
  } catch {
    // older @tauri-apps/api or a non-browser environment
  }
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

let instance: Promise<Backend> | null = null;

/** The Tauri backend inside the desktop shell, otherwise the scripted mock (for `pnpm dev` in a browser). */
export function getBackend(): Promise<Backend> {
  if (!instance) {
    instance = runningInTauri()
      ? import("./tauri").then((m) => new m.TauriBackend())
      : import("./mock").then((m) => new m.MockBackend());
  }
  return instance;
}
