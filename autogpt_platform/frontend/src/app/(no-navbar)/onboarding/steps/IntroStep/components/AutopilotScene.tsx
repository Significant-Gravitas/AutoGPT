import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AUTOPILOT_AVATAR } from "@/components/molecules/BotAvatar/helpers";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { motion } from "framer-motion";

const POINTS = [
  "I manage everything for you.",
  "When an expert gets stuck, I step in.",
  "I delegate tasks across the team.",
  "I hire new experts when you need them.",
];

const EASE = [0.22, 1, 0.36, 1] as const;

function reveal(i: number) {
  return { delay: 0.25 + i * 0.15, duration: 0.5, ease: EASE };
}

export function AutopilotScene() {
  return (
    <div className="flex h-full flex-col items-center justify-start gap-4">
      <motion.div
        initial={{ opacity: 0, y: 12, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: EASE }}
      >
        <BotAvatar
          config={AUTOPILOT_AVATAR}
          status="done"
          expression="big"
          size={120}
          trackPointer
          showBadge={false}
        />
      </motion.div>

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
