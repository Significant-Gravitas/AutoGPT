import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { motion } from "framer-motion";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";

const POINTS = [
  "I help plan and coordinate your work.",
  "When an expert gets stuck, I step in.",
  "I delegate tasks across the team.",
  "I suggest Experts you can hire.",
];

const EASE = [0.22, 1, 0.36, 1] as const;

function reveal(i: number) {
  return { delay: 0.25 + i * 0.15, duration: 0.5, ease: EASE };
}

export function AutopilotScene() {
  return (
    <div className="flex h-full flex-col items-center justify-start gap-4">
      <AutopilotAvatar size={120} transparent />

      <div className="flex flex-col items-center gap-3">
        <ul className="flex flex-col items-center gap-2">
          {POINTS.map((point, i) => (
            <motion.li
              key={point}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={reveal(i)}
              className="flex items-center gap-1.5"
            >
              <Icon
                icon={ArrowRight01Icon}
                size={16}
                className="shrink-0 text-purple-500"
              />
              <Text variant="body" as="span">
                {point}
              </Text>
            </motion.li>
          ))}
        </ul>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={reveal(POINTS.length)}
        >
          <Text variant="body" as="p" tone="muted">
            and more…
          </Text>
        </motion.div>
      </div>
    </div>
  );
}
