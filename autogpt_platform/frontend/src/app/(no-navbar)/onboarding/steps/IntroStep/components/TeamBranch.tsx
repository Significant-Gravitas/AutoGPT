import { motion } from "framer-motion";
import type { CSSProperties } from "react";
import {
  branchDelay,
  stagePercent,
  teamBranch,
  type TeamMember,
} from "./teamSceneHelpers";

const EASE = [0.22, 1, 0.36, 1] as const;
const LINE = "absolute border-border-base";

interface Props {
  parent: TeamMember;
  animate: boolean;
}

// Drawn with borders rather than an SVG, so the lines stay 2px and the bar
// keeps round corners however far the stage stretches.
export function TeamBranch({ parent, animate }: Props) {
  const branch = teamBranch(parent);
  const delay = branchDelay(parent);
  const lines = {
    "--x": stagePercent(branch.x),
    "--start": `${branch.start}px`,
    "--middle": `${branch.middle}px`,
    "--end": `${branch.end}px`,
    "--left": stagePercent(branch.left),
    "--right": stagePercent(branch.right),
  } as CSSProperties;

  return (
    <div aria-hidden="true" style={lines} className="absolute inset-0">
      <motion.div
        className={`${LINE} top-(--start) left-[calc(var(--x)-1px)] h-[calc(var(--middle)-var(--start))] origin-top border-l-2`}
        initial={animate ? { scaleY: 0 } : false}
        animate={{ scaleY: 1 }}
        transition={{ delay, duration: 0.25, ease: EASE }}
      />
      <motion.div
        className={`${LINE} top-(--middle) left-[calc(var(--left)-1px)] h-[calc(var(--end)-var(--middle))] w-[calc(var(--right)-var(--left)+2px)] rounded-t-xl border-x-2 border-t-2`}
        initial={animate ? { opacity: 0, scaleX: 0.4 } : false}
        animate={{ opacity: 1, scaleX: 1 }}
        transition={{ delay: delay + 0.15, duration: 0.4, ease: EASE }}
      />
      {branch.innerDrops.map((x) => (
        <motion.div
          key={x}
          style={{ "--drop": stagePercent(x) } as CSSProperties}
          className={`${LINE} top-(--middle) left-[calc(var(--drop)-1px)] h-[calc(var(--end)-var(--middle))] origin-top border-l-2`}
          initial={animate ? { scaleY: 0 } : false}
          animate={{ scaleY: 1 }}
          transition={{ delay: delay + 0.25, duration: 0.25, ease: EASE }}
        />
      ))}
    </div>
  );
}
