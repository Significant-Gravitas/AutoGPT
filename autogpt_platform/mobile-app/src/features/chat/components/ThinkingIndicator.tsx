import Animated from "react-native-reanimated";

import { createStyles, radius } from "@/lib/theme";

const SIZE = 12;

/** A softly pulsing dot while the reply has not started. */
export function ThinkingIndicator() {
  const styles = useStyles();
  return (
    <Animated.View
      accessible
      accessibilityLabel="Thinking"
      accessibilityLiveRegion="polite"
      style={[styles.dot, pulse]}
    />
  );
}

const pulse = {
  animationName: {
    from: { opacity: 0.45, transform: [{ scale: 0.8 }] },
    to: { opacity: 1, transform: [{ scale: 1 }] },
  },
  animationDuration: "700ms",
  animationIterationCount: "infinite",
  animationDirection: "alternate",
  animationTimingFunction: "ease-in-out",
} as const;

const useStyles = createStyles((colors) => ({
  dot: {
    width: SIZE,
    height: SIZE,
    marginVertical: 6,
    borderRadius: radius.pill,
    backgroundColor: colors.text,
  },
}));
