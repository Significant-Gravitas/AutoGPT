import { useState } from "react";
import { FlatList, Platform, Pressable, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, radius, spacing } from "@/lib/theme";

const MONO = Platform.select({ ios: "Menlo", default: "monospace" });
const LINE_HEIGHT = 20;

interface Props {
  text: string;
  cappedNotice: string | null;
}

/**
 * Code and other text with line numbers. Unwrapped, it scrolls sideways and
 * the numbers stay in step; wrapped, each line is its own row so a long one
 * keeps its number beside it.
 */
export function CodeView({ text, cappedNotice }: Props) {
  const styles = useStyles();
  const insets = useSafeAreaInsets();
  const [isWrapped, setIsWrapped] = useState(false);
  const lines = text.split("\n");
  const gutter = String(lines.length).length * 8 + 16;
  const bottom = { paddingBottom: insets.bottom + spacing.xl };

  function toggleWrap() {
    setIsWrapped(!isWrapped);
  }

  return (
    <View style={styles.screen}>
      <View style={styles.bar}>
        <Pressable
          accessibilityRole="switch"
          accessibilityState={{ checked: isWrapped }}
          onPress={toggleWrap}
          style={[styles.chip, isWrapped && styles.chipOn]}
        >
          <Text style={[styles.chipLabel, isWrapped && styles.chipLabelOn]}>Wrap lines</Text>
        </Pressable>
        {cappedNotice ? <Text style={styles.capped}>{cappedNotice}</Text> : null}
      </View>
      {isWrapped ? (
        <FlatList
          data={lines}
          keyExtractor={(_, index) => String(index)}
          contentContainerStyle={bottom}
          renderItem={({ item, index }) => (
            <View style={styles.row}>
              <Text style={[styles.number, { width: gutter }]}>{index + 1}</Text>
              <Text selectable style={[styles.code, styles.wrapped]}>
                {item || " "}
              </Text>
            </View>
          )}
        />
      ) : (
        <ScrollView contentContainerStyle={bottom}>
          <View style={styles.row}>
            <Text style={[styles.number, { width: gutter }]}>
              {lines.map((_, index) => index + 1).join("\n")}
            </Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <Text selectable style={styles.code}>
                {text}
              </Text>
            </ScrollView>
          </View>
        </ScrollView>
      )}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1 },
  bar: { flexDirection: "row", alignItems: "center", gap: spacing.sm, paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  chip: { paddingHorizontal: 11, height: 28, borderRadius: radius.pill, justifyContent: "center", backgroundColor: colors.surfaceRaised },
  chipOn: { backgroundColor: colors.action },
  chipLabel: { fontSize: 12.5, fontWeight: "500", color: colors.textSecondary },
  chipLabelOn: { color: colors.onAction },
  capped: { flex: 1, fontSize: 12, color: colors.textTertiary },
  row: { flexDirection: "row", alignItems: "flex-start" },
  number: {
    fontFamily: MONO,
    fontSize: 12.5,
    lineHeight: LINE_HEIGHT,
    color: colors.textTertiary,
    textAlign: "right",
    paddingRight: 10,
  },
  code: { fontFamily: MONO, fontSize: 12.5, lineHeight: LINE_HEIGHT, color: colors.text, paddingRight: spacing.lg },
  wrapped: { flex: 1 },
}));
