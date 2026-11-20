import { Menu01Icon } from "@hugeicons/core-free-icons";
import { useNavigation } from "expo-router";
import { DrawerActions } from "expo-router/react-navigation";
import type { ReactNode } from "react";
import { View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import Svg, { Defs, LinearGradient, Rect, Stop } from "react-native-svg";

import { createStyles, spacing, useColors } from "@/lib/theme";

import { IconButton } from "./IconButton";

/** The header row's height, under the status bar. */
export const FLOATING_HEADER_HEIGHT = 52;

// How far the fade reaches past the header's foot.
const FADE_TAIL = 20;

interface Props {
  /** In place of the menu button, e.g. a back button. */
  left?: ReactNode;
  center?: ReactNode;
  right?: ReactNode;
  /** The colour the fade starts from; the page's background by default. */
  fadeColor?: string;
}

/**
 * A header that floats over the page, as in the ChatGPT app: round buttons
 * on a fade that runs from the page colour to clear, so the content scrolls
 * out under it rather than meeting a hard edge. The page pads its own content
 * by `insets.top + FLOATING_HEADER_HEIGHT`.
 */
export function FloatingHeader({ left, center, right, fadeColor }: Props) {
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();
  const fadeHeight = insets.top + FLOATING_HEADER_HEIGHT + FADE_TAIL;
  const fade = fadeColor ?? colors.background;

  function openMenu() {
    navigation.dispatch(DrawerActions.openDrawer());
  }

  return (
    <View style={styles.wrap}>
      <View style={[styles.fade, { height: fadeHeight }]}>
        <Svg width="100%" height="100%" preserveAspectRatio="none">
          <Defs>
            <LinearGradient id="headerFade" x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0" stopColor={fade} stopOpacity={1} />
              <Stop offset="0.6" stopColor={fade} stopOpacity={1} />
              <Stop offset="0.8" stopColor={fade} stopOpacity={0.5} />
              <Stop offset="1" stopColor={fade} stopOpacity={0} />
            </LinearGradient>
          </Defs>
          <Rect width="100%" height="100%" fill="url(#headerFade)" />
        </Svg>
      </View>
      <View style={[styles.row, { marginTop: insets.top }]}>
        <View style={styles.side}>
          {left ?? (
            <IconButton
              icon={Menu01Icon}
              label="Open menu"
              onPress={openMenu}
              variant="floating"
            />
          )}
        </View>
        <View style={styles.center}>{center}</View>
        <View style={[styles.side, styles.right]}>{right}</View>
      </View>
    </View>
  );
}

const useStyles = createStyles(() => ({
  // Taps between the buttons fall through to the content underneath.
  wrap: { position: "absolute", top: 0, left: 0, right: 0, pointerEvents: "box-none" },
  fade: { position: "absolute", top: 0, left: 0, right: 0, pointerEvents: "none" },
  row: {
    height: FLOATING_HEADER_HEIGHT,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    pointerEvents: "box-none",
  },
  center: { flex: 1, alignItems: "center", pointerEvents: "box-none" },
  // One width on both sides keeps the centre centred.
  side: { width: 38, flexDirection: "row" },
  right: { justifyContent: "flex-end" },
}));
