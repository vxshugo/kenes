import { useEffect, useRef } from "react";
import { controller } from "../session/useController";
import type { Source } from "../types";

/** rms → 0..1 on a dB scale (-55 dBFS … -5 dBFS). */
function toLevel(rms: number): number {
  if (rms <= 0) return 0;
  const db = 20 * Math.log10(Math.max(rms, 1e-6));
  return Math.min(1, Math.max(0, (db + 55) / 50));
}

/** Level bar fed straight from the controller, animated outside React renders. */
export function LevelMeter({ source, label, disabled }: { source: Source; label: string; disabled?: boolean }) {
  const barRef = useRef<HTMLSpanElement>(null);
  const target = useRef(0);
  const shown = useRef(0);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      // Fast attack, slow release; the loop parks itself once the bar is back at zero.
      const t = target.current;
      shown.current = t > shown.current ? shown.current + (t - shown.current) * 0.6 : shown.current * 0.88 + t * 0.12;
      if (t === 0 && shown.current < 0.002) shown.current = 0;
      if (barRef.current) barRef.current.style.transform = `scaleX(${shown.current.toFixed(3)})`;
      raf = shown.current > 0 || t > 0 ? requestAnimationFrame(tick) : 0;
    };
    const off = controller.subscribeLevels((l) => {
      target.current = toLevel(l[source]);
      if (!raf) raf = requestAnimationFrame(tick);
    });
    return () => {
      off();
      cancelAnimationFrame(raf);
    };
  }, [source]);

  return (
    <span className={`meter meter-${source}${disabled ? " is-off" : ""}`} title={disabled ? `${label}: захват выключен` : `Уровень: ${label}`}>
      <span className="meter-label">{label}</span>
      <span className="meter-track" aria-hidden="true">
        <span className="meter-bar" ref={barRef} />
      </span>
    </span>
  );
}
