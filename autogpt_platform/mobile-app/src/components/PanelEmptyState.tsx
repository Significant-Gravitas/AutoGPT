import type { ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";

import { createStyles, spacing, useScheme } from "@/lib/theme";

interface Props {
  /** Left, right and centre tile, in that order. */
  tiles: [ReactNode, ReactNode, ReactNode];
  title: string;
  description: string;
  children?: ReactNode;
}

/** The side panel's empty state, as on the web: three tiles, the outer two
 *  tilted out, over a title and a line of copy. */
export function PanelEmptyState({ tiles, title, description, children }: Props) {
  const styles = useStyles();
  const ring = useScheme() === "dark" ? styles.darkRing : styles.lightRing;
  const positions = [styles.left, styles.right, styles.centre];

  return (
    <View style={styles.container}>
      <View
        accessibilityElementsHidden
        importantForAccessibility="no-hide-descendants"
        style={styles.stack}
      >
        {tiles.map((tile, index) => (
          <View key={index} style={[styles.tile, ring, positions[index]]}>
            {tile}
          </View>
        ))}
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.description}>{description}</Text>
      {children ? <View style={styles.actions}>{children}</View> : null}
    </View>
  );
}

const TILE_WIDTH = 64;

const useStyles = createStyles((colors) => ({
  container: {
    alignItems: "center",
    paddingHorizontal: spacing.xl,
    paddingTop: 40,
    paddingBottom: spacing.xxl,
  },
  stack: { width: 144, height: 96, marginBottom: 28 },
  tile: {
    position: "absolute",
    width: TILE_WIDTH,
    height: 80,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
  // The web's smooth-shadow-ring-sm: a faint ring plus a soft shadow.
  lightRing: {
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.05)",
    shadowColor: "#000000",
    shadowOpacity: 0.06,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  darkRing: { borderWidth: StyleSheet.hairlineWidth, borderColor: colors.border },
  left: { left: 8, top: 12, transform: [{ rotate: "-12deg" }] },
  right: { right: 8, top: 12, transform: [{ rotate: "12deg" }] },
  centre: { left: (144 - TILE_WIDTH) / 2, top: 0, zIndex: 10 },
  title: {
    fontSize: 18,
    lineHeight: 24,
    fontWeight: "500",
    color: colors.text,
    textAlign: "center",
  },
  description: {
    marginTop: 6,
    maxWidth: 272,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textSecondary,
    textAlign: "center",
  },
  actions: {
    marginTop: 20,
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: spacing.sm,
  },
}));
