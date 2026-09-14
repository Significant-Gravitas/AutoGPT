import { useEffect, useState } from "react";
import {
  BLINK_CADENCE_MS,
  BLINK_EYES,
  EXPRESSION_CADENCE_MS,
  EXPRESSION_POOLS,
  type AvatarStatus,
  type Expression,
} from "./expressions";

const BLINK_MS = 130;

function between([min, max]: [number, number]) {
  return min + Math.random() * (max - min);
}

interface Args {
  status: AvatarStatus;
  isLive: boolean;
  override?: Expression;
}

export function useExpression({ status, isLive, override }: Args) {
  const [expression, setExpression] = useState<Expression>(
    EXPRESSION_POOLS[status][0],
  );
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    setExpression(EXPRESSION_POOLS[status][0]);
    if (!isLive) return;
    let timer: ReturnType<typeof setTimeout>;
    function schedule() {
      timer = setTimeout(() => {
        setExpression((current) => {
          const pool = EXPRESSION_POOLS[status];
          const alternatives = pool.filter((entry) => entry !== current);
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
    setIsBlinking(false);
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

  const resolved = override ?? expression;
  return {
    expression: isBlinking ? { ...resolved, eyes: BLINK_EYES } : resolved,
    isBlinking,
  };
}
