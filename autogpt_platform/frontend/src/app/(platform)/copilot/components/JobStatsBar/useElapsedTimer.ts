import { useEffect, useRef, useState } from "react";

export function useElapsedTimer(isRunning: boolean) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    if (isRunning) {
      if (startTimeRef.current === null) {
        startTimeRef.current = Date.now();
        setElapsedSeconds(0);
      }

      intervalRef.current = setInterval(() => {
        if (startTimeRef.current !== null) {
          setElapsedSeconds(
            Math.floor((Date.now() - startTimeRef.current) / 1000),
          );
        }
      }, 1000);

      return () => clearInterval(intervalRef.current);
    }

    clearInterval(intervalRef.current);
    startTimeRef.current = null;
  }, [isRunning]);

  return { elapsedSeconds };
}
