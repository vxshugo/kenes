import { useEffect, useState } from "react";
import { formatElapsed } from "../lib/format";

export function Elapsed({ startedAt, endedAt }: { startedAt: number | null; endedAt: number | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!startedAt || endedAt) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [startedAt, endedAt]);
  if (!startedAt) return <span className="elapsed muted">00:00</span>;
  return (
    <span className="elapsed" aria-label="Длительность">
      {formatElapsed((endedAt ?? now) - startedAt)}
    </span>
  );
}
