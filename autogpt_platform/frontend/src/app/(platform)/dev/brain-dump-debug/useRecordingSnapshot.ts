import { useEffect, useState } from "react";
import {
  EMPTY_SNAPSHOT,
  readRecordingSnapshot,
  SNAPSHOT_POLL_MS,
  type RecordingSnapshot,
} from "./helpers";

export function useRecordingSnapshot() {
  const [snapshot, setSnapshot] = useState<RecordingSnapshot>(EMPTY_SNAPSHOT);

  useEffect(() => {
    let isActive = true;

    async function poll() {
      const next = await readRecordingSnapshot();
      if (isActive) setSnapshot(next);
    }

    void poll();
    const timer = setInterval(function tick() {
      void poll();
    }, SNAPSHOT_POLL_MS);

    return () => {
      isActive = false;
      clearInterval(timer);
    };
  }, []);

  async function refresh() {
    setSnapshot(await readRecordingSnapshot());
  }

  return { snapshot, refresh };
}
