import { Icon } from "@/components/atoms/Icon/Icon";
import type { Tone } from "@/components/styles/tones";
import {
  AlertSquareIcon,
  SignalFull02Icon,
  SignalLow02Icon,
  SignalMedium02Icon,
  SignalNo02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { PRIORITY_LABELS } from "../../data/helpers";
import type { TaskPriority } from "../../data/types";

const GLYPHS: Record<TaskPriority, { icon: IconSvgElement; tone: Tone }> = {
  urgent: { icon: AlertSquareIcon, tone: "error" },
  high: { icon: SignalFull02Icon, tone: "emphasis" },
  medium: { icon: SignalMedium02Icon, tone: "secondary" },
  low: { icon: SignalLow02Icon, tone: "muted" },
  none: { icon: SignalNo02Icon, tone: "subtle" },
};

interface Props {
  priority: TaskPriority;
  size?: number;
}

export function PriorityGlyph({ priority, size = 14 }: Props) {
  const glyph = GLYPHS[priority];
  return (
    <span
      className="inline-flex shrink-0"
      role="img"
      aria-label={PRIORITY_LABELS[priority]}
      title={PRIORITY_LABELS[priority]}
    >
      <Icon icon={glyph.icon} size={size} tone={glyph.tone} />
    </span>
  );
}
