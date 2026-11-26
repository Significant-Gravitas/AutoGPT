import { File02Icon, Image01Icon, Table01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";
import { StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, useColors, useScheme } from "@/lib/theme";

interface Props {
  /** Set in an expert's chat, where the list is that expert's documents. */
  expertName?: string;
}

/**
 * A short card in place of the files list: the panel's three tiles, small,
 * beside the web's wording, so the sections under it stay on screen.
 */
export function FilesEmptyState({ expertName }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const ring = useScheme() === "dark" ? styles.darkRing : styles.lightRing;
  const tiles: [IconSvgElement, string, object][] = [
    [Image01Icon, colors.sky, styles.left],
    [Table01Icon, colors.success, styles.right],
    [File02Icon, colors.brand, styles.centre],
  ];

  return (
    <View style={styles.card}>
      <View
        accessibilityElementsHidden
        importantForAccessibility="no-hide-descendants"
        style={styles.stack}
      >
        {tiles.map(([icon, color, position], index) => (
          <View key={index} style={[styles.tile, ring, position]}>
            <Icon icon={icon} size={16} color={color} />
          </View>
        ))}
      </View>
      <View style={styles.text}>
        <Text style={styles.title}>No files yet</Text>
        <Text style={styles.description}>
          {expertName
            ? `Files you share with ${expertName} and documents it creates show up here.`
            : "Files you attach and documents Otto creates in this chat show up here."}
        </Text>
      </View>
    </View>
  );
}

const STACK_WIDTH = 66;
const TILE_WIDTH = 36;

const useStyles = createStyles((colors) => ({
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.lg,
    paddingHorizontal: 18,
    paddingVertical: 20,
    borderRadius: radius.lg,
    backgroundColor: colors.panelCard,
  },
  stack: { width: STACK_WIDTH, height: 58 },
  tile: {
    position: "absolute",
    width: TILE_WIDTH,
    height: 46,
    borderRadius: 9,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.panelCard,
  },
  lightRing: {
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.06)",
    shadowColor: "#000000",
    shadowOpacity: 0.07,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  darkRing: { borderWidth: StyleSheet.hairlineWidth, borderColor: colors.border },
  left: { left: 2, top: 8, transform: [{ rotate: "-12deg" }] },
  right: { right: 2, top: 8, transform: [{ rotate: "12deg" }] },
  centre: { left: (STACK_WIDTH - TILE_WIDTH) / 2, top: 2, zIndex: 10 },
  text: { flex: 1, minWidth: 0, gap: 3 },
  title: { fontSize: 16, lineHeight: 21, fontWeight: "600", color: colors.text },
  description: { fontSize: 14, lineHeight: 19, color: colors.textSecondary },
}));
