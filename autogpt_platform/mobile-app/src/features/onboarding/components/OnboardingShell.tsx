import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import { createStyles, radius, spacing, type } from "@/lib/theme";

const STEP_COUNT = 3;

interface Props {
  /** How many of the three bars are filled. */
  filled: number;
  onBack?: () => void;
  onSkip?: () => void;
  footer: ReactNode;
  children: ReactNode;
}

/** An onboarding step: back, progress and Skip on top, the footer pinned at the bottom. */
export function OnboardingShell({ filled, onBack, onSkip, footer, children }: Props) {
  const styles = useStyles();
  const insets = useSafeAreaInsets();

  return (
    <KeyboardAvoidingView
      behavior="padding"
      style={[styles.screen, { paddingTop: insets.top }]}
    >
      <View style={styles.top}>
        <View style={styles.side}>
          {onBack ? (
            <IconButton icon={ArrowLeft01Icon} label="Back" onPress={onBack} />
          ) : null}
        </View>
        <View
          accessibilityRole="progressbar"
          accessibilityLabel={`Step ${filled} of ${STEP_COUNT}`}
          style={styles.progress}
        >
          {Array.from({ length: STEP_COUNT }, (_, index) => (
            <View
              key={index}
              style={[styles.bar, index < filled && styles.barFilled]}
            />
          ))}
        </View>
        <View style={[styles.side, styles.sideEnd]}>
          {onSkip ? (
            <Pressable
              accessibilityRole="button"
              hitSlop={8}
              onPress={onSkip}
              style={({ pressed }) => [styles.skip, pressed && styles.pressed]}
            >
              <Text style={styles.skipLabel}>Skip</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
      <ScrollView
        style={styles.body}
        contentContainerStyle={styles.bodyContent}
        keyboardShouldPersistTaps="handled"
      >
        {children}
      </ScrollView>
      <View
        style={[styles.footer, { paddingBottom: insets.bottom + spacing.md }]}
      >
        {footer}
      </View>
    </KeyboardAvoidingView>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  top: {
    height: 44,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.md,
  },
  side: { width: 72, flexDirection: "row" },
  sideEnd: { justifyContent: "flex-end" },
  progress: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "center",
    gap: 6,
  },
  bar: {
    width: 28,
    height: 4,
    borderRadius: radius.pill,
    backgroundColor: colors.surfacePressed,
  },
  barFilled: { backgroundColor: colors.text },
  skip: { padding: spacing.sm },
  skipLabel: { ...type.label, color: colors.textSecondary },
  pressed: { opacity: 0.6 },
  body: { flex: 1 },
  bodyContent: {
    flexGrow: 1,
    alignItems: "center",
    paddingTop: spacing.sm,
    paddingHorizontal: spacing.xl,
  },
  footer: {
    gap: spacing.md,
    paddingTop: spacing.md,
    paddingHorizontal: spacing.xl,
  },
}));
