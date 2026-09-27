import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, normalizeSettings, parseMyNames } from "../types";
import { applyFormat, formatOf, MEETING_FORMATS } from "./meetingFormat";

describe("settings migration", () => {
  it("fills new keys with defaults", () => {
    const s = normalizeSettings({});
    expect(s.micMode).toBe("me");
    expect(s.myNames).toEqual([]);
    expect(s.autoHintMode).toBe("any");
    expect("autoHints" in s).toBe(false);
  });

  it("autoHints: false becomes \"off\"; true or missing follows the mic mode", () => {
    expect(normalizeSettings({ autoHints: false }).autoHintMode).toBe("off");
    expect(normalizeSettings({ autoHints: false, micMode: "room" }).autoHintMode).toBe("off");
    expect(normalizeSettings({ autoHints: true }).autoHintMode).toBe("any");
    expect(normalizeSettings({ autoHints: true, micMode: "room" }).autoHintMode).toBe("addressed");
    expect(normalizeSettings({ micMode: "room" }).autoHintMode).toBe("addressed");
  });

  it("an explicit autoHintMode wins over the legacy flag (Rust defaults keep sending autoHints: true)", () => {
    expect(normalizeSettings({ autoHints: true, autoHintMode: "off" }).autoHintMode).toBe("off");
    expect(normalizeSettings({ autoHints: false, autoHintMode: "addressed" }).autoHintMode).toBe("addressed");
    expect(normalizeSettings({ autoHintMode: "sometimes" }).autoHintMode).toBe("any");
  });

  it("new keys: echo cancellation, GNOME always-on-top and global shortcuts default on", () => {
    const s = normalizeSettings({});
    expect(s.echoCancellation).toBe(true);
    expect(s.gnomeAlwaysOnTop).toBe(true);
    expect(s.globalHotkeys).toBe(true);
    expect(s.hotkeys).toEqual({ hint: "CommandOrControl+Alt+Enter", recap: "CommandOrControl+Alt+KeyK", toggle: "CommandOrControl+Alt+KeyP" });
    const off = normalizeSettings({ echoCancellation: false, gnomeAlwaysOnTop: false, globalHotkeys: false });
    expect([off.echoCancellation, off.gnomeAlwaysOnTop, off.globalHotkeys]).toEqual([false, false, false]);
    expect(normalizeSettings({ echoCancellation: "yes" }).echoCancellation).toBe(true);
  });

  it("drops the legacy key but keeps other unknown keys", () => {
    const s = normalizeSettings({ autoHints: false, rustOnly: { x: 1 } }) as unknown as Record<string, unknown>;
    expect(s.autoHints).toBeUndefined();
    expect(s.rustOnly).toEqual({ x: 1 });
  });

  it("validates micMode and myNames", () => {
    expect(normalizeSettings({ micMode: "crowd" }).micMode).toBe("me");
    expect(normalizeSettings({ myNames: "Хуго, Hugo ; хуго,, " }).myNames).toEqual(["Хуго", "Hugo"]);
    expect(normalizeSettings({ myNames: ["  Хуго ", 5, "", "Hugo"] }).myNames).toEqual(["Хуго", "Hugo"]);
    expect(parseMyNames(Array.from({ length: 20 }, (_, i) => `N${i}`))).toHaveLength(10);
  });
});

describe("meeting formats", () => {
  it("map to capture settings and back", () => {
    expect(MEETING_FORMATS.online).toMatchObject({ captureMic: true, captureSystem: true, micMode: "me" });
    expect(MEETING_FORMATS.room).toMatchObject({ captureMic: true, captureSystem: false, micMode: "room" });
    expect(MEETING_FORMATS.hybrid).toMatchObject({ captureMic: true, captureSystem: true, micMode: "room" });
    expect(formatOf(DEFAULT_SETTINGS)).toBe("online");
    expect(formatOf({ captureMic: true, captureSystem: false, micMode: "me" })).toBeNull();
  });

  it("apply the auto-hint default of the new mic mode unless the user picked another mode", () => {
    const room = applyFormat(DEFAULT_SETTINGS, "room");
    expect(room).toMatchObject({ captureSystem: false, micMode: "room", autoHintMode: "addressed" });
    expect(applyFormat(room, "online").autoHintMode).toBe("any");
    expect(applyFormat({ ...DEFAULT_SETTINGS, autoHintMode: "off" }, "hybrid").autoHintMode).toBe("off");
    expect(applyFormat({ ...DEFAULT_SETTINGS, autoHintMode: "addressed" }, "room").autoHintMode).toBe("addressed");
    expect(applyFormat({ ...room, autoHintMode: "any" }, "online").autoHintMode).toBe("any");
  });
});
