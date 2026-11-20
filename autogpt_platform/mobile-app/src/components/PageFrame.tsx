import { use, type ReactNode } from "react";
import { StyleSheet } from "react-native";
import { DrawerProgressContext } from "react-native-drawer-layout";
import Animated, { useAnimatedStyle, type SharedValue } from "react-native-reanimated";

import { createStyles, radius } from "@/lib/theme";

const PAGE_RADIUS = radius.page;
const SHADOW_OPACITY = 0.14;
const ELEVATION = 16;

interface Props {
  children: ReactNode;
}

/**
 * A page inside a drawer. As the drawer opens, the page rounds its corners,
 * lifts on a soft shadow and fades under a wash, as in the ChatGPT app.
 * The drawer's own see-through overlay above it still takes the tap that
 * closes the drawer.
 */
export function PageFrame({ children }: Props) {
  const progress = use(DrawerProgressContext);
  // Outside a drawer (a navigator's fallback screen) there is nothing to frame.
  if (!progress) return children;
  return <AnimatedFrame progress={progress}>{children}</AnimatedFrame>;
}

interface FrameProps {
  progress: Readonly<SharedValue<number>>;
  children: ReactNode;
}

function AnimatedFrame({ progress, children }: FrameProps) {
  const styles = useStyles();

  // The outer view casts the shadow, so it must not clip.
  const frameStyle = useAnimatedStyle(() => ({
    borderRadius: progress.value * PAGE_RADIUS,
    shadowOpacity: progress.value * SHADOW_OPACITY,
    elevation: progress.value * ELEVATION,
  }));
  const clipStyle = useAnimatedStyle(() => ({
    borderRadius: progress.value * PAGE_RADIUS,
  }));
  const washStyle = useAnimatedStyle(() => ({
    borderRadius: progress.value * PAGE_RADIUS,
    opacity: progress.value,
  }));

  return (
    <Animated.View style={[styles.frame, frameStyle]}>
      <Animated.View style={[styles.clip, clipStyle]}>
        {children}
        <Animated.View style={[styles.wash, washStyle]} />
      </Animated.View>
    </Animated.View>
  );
}

const useStyles = createStyles((colors) => ({
  frame: {
    flex: 1,
    backgroundColor: colors.background,
    shadowColor: "#000000",
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 0 },
  },
  clip: { flex: 1, overflow: "hidden" },
  wash: {
    ...StyleSheet.absoluteFill,
    backgroundColor: colors.pageWash,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.pageEdge,
    pointerEvents: "none",
  },
}));
