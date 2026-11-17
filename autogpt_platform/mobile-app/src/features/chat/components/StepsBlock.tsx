import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { stepIcons, type Step } from "../steps";
import { StepRow } from "./StepRow";

const TILE = 24;
const OVERLAP = 7;

interface Props {
  steps: Step[];
  isLive?: boolean;
}

export function StepsBlock({ steps, isLive = false }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const [isOpen, setIsOpen] = useState(false);
  const live = isLive || steps.some((step) => step.state === "running");

  if (live) return <Rail steps={steps} />;

  const count = steps.length === 1 ? "1 step" : `${steps.length} steps`;

  function handlePress() {
    setIsOpen((open) => !open);
  }

  return (
    <View style={styles.block}>
      <Pressable
        accessibilityRole="button"
        accessibilityState={{ expanded: isOpen }}
        accessibilityLabel={count}
        onPress={handlePress}
        hitSlop={spacing.xs}
        style={({ pressed }) => [styles.header, pressed && styles.pressed]}
      >
        <View style={styles.tiles}>
          {steps.slice(0, 3).map((step, index) => (
            <View
              key={step.id}
              style={[styles.headerTile, index > 0 && styles.overlap]}
            >
              <Icon
                icon={stepIcons[step.kind]}
                size={12}
                color={colors.textSecondary}
              />
            </View>
          ))}
        </View>
        <Text style={styles.count}>{count}</Text>
        <Icon
          icon={isOpen ? ArrowUp01Icon : ArrowDown01Icon}
          size={14}
          color={colors.textTertiary}
        />
      </Pressable>
      {isOpen ? <Rail steps={steps} /> : null}
    </View>
  );
}

interface RailProps {
  steps: Step[];
}

function Rail({ steps }: RailProps) {
  const styles = useStyles();
  return (
    <View style={styles.rail}>
      {steps.length > 1 ? <View style={styles.line} /> : null}
      {steps.map((step) => (
        <StepRow key={step.id} step={step} />
      ))}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  block: { gap: spacing.sm + 2 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: spacing.sm,
    borderRadius: 7,
  },
  pressed: { opacity: 0.6 },
  tiles: { flexDirection: "row" },
  headerTile: {
    width: TILE,
    height: TILE,
    borderRadius: 7,
    borderWidth: 2,
    borderColor: colors.background,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.surfaceRaised,
  },
  overlap: { marginLeft: -OVERLAP },
  count: {
    fontSize: 13,
    fontWeight: "500",
    lineHeight: 18,
    color: colors.textSecondary,
  },
  rail: { gap: spacing.sm + 2 },
  line: {
    position: "absolute",
    left: TILE / 2 - 0.5,
    top: TILE / 2,
    bottom: TILE / 2,
    width: 1,
    backgroundColor: colors.border,
  },
}));
