import { useEffect, useState } from "react";
import {
  BLINK_CADENCE_MS,
  EXPRESSION_CADENCE_MS,
  POOLS,
  type ExpressionId,
} from "./expressions";
import type { AvatarStatus } from "./helpers";

const BLINK_MS = 130;

function between([min, max]: [number, number]) {
  return min + Math.random() * (max - min);
}

interface Args {
  status: AvatarStatus;
  isLive: boolean;
  override?: ExpressionId;
}

export function useExpression({ status, isLive, override }: Args) {
  const [expression, setExpression] = useState<ExpressionId>(POOLS[status][0]);
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    setExpression(POOLS[status][0]);
    if (!isLive) return;
    let timer: ReturnType<typeof setTimeout>;
    function schedule() {
      timer = setTimeout(() => {
        setExpression((current) => {
          const alternatives = POOLS[status].filter((id) => id !== current);
          return (
            alternatives[Math.floor(Math.random() * alternatives.length)] ??
            current
          );
        });
        schedule();
      }, between(EXPRESSION_CADENCE_MS[status]));
    }
    schedule();
    return () => clearTimeout(timer);
  }, [status, isLive]);

  useEffect(() => {
    const cadence = BLINK_CADENCE_MS[status];
    if (!isLive || !cadence) return;
    const range: [number, number] = cadence;
    let timer: ReturnType<typeof setTimeout>;
    function schedule() {
      timer = setTimeout(() => {
        setIsBlinking(true);
        timer = setTimeout(() => {
          setIsBlinking(false);
          schedule();
        }, BLINK_MS);
      }, between(range));
    }
    schedule();
    return () => clearTimeout(timer);
  }, [status, isLive]);

  return { expression: override ?? expression, isBlinking };
}
