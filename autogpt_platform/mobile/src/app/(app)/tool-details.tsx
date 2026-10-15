import { router, useLocalSearchParams } from "expo-router";
import { ScrollView, StyleSheet, Text, View } from "react-native";

import { PlainButton } from "../../design/components/Buttons";
import { Row, Section } from "../../design/components/Grouped";
import { Icon } from "../../design/components/Icon";
import { EmptyState } from "../../design/components/StateViews";
import { colors, radius, spacing, type } from "../../design/theme";
import { recallToolPart } from "../../features/chat/tools/detailsStore";
import { presentTool } from "../../features/chat/tools/presentation";
import { boundedPrettyPrint } from "../../lib/json";

/**
 * Read-only details for one tool step: the bounded, secret-masked input and
 * result, and a plain statement of what the app can't do with it yet.
 */
export default function ToolDetailsScreen() {
  const { key } = useLocalSearchParams<{ key: string }>();
  const part = key ? recallToolPart(key) : null;

  if (!part) {
    return (
      <EmptyState
        symbol="info.circle"
        title="Details unavailable"
        message="Open the step again from its conversation."
        action={<PlainButton title="Close" onPress={() => router.back()} />}
      />
    );
  }

  const presentation = presentTool(part);
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Icon name={presentation.symbol} size={22} color={colors.tint} />
        <Text
          style={[type.headline, styles.title]}
          accessibilityRole="header"
          numberOfLines={2}
        >
          {presentation.title}
        </Text>
        <PlainButton title="Done" onPress={() => router.back()} />
      </View>
      <Section>
        <Row
          title="Status"
          trailing={
            <Text style={[type.body, styles.value]}>
              {presentation.statusLabel}
            </Text>
          }
        />
        <Row
          title="Tool"
          trailing={
            <Text style={[type.body, styles.value]}>
              {part.toolName || "Unknown"}
            </Text>
          }
        />
        <Row
          title="Result type"
          trailing={
            <Text style={[type.body, styles.value]}>
              {presentation.responseType ?? "None"}
            </Text>
          }
        />
        <Row
          title="Card"
          trailing={
            <Text style={[type.body, styles.value]}>
              {presentation.family?.label ?? "Not recognized"}
            </Text>
          }
        />
      </Section>
      <Text style={[type.footnote, styles.note]}>
        {presentation.coverageNote}
      </Text>
      {presentation.summary ? (
        <Section header="Summary">
          <Text selectable style={[type.body, styles.summary]}>
            {presentation.summary}
          </Text>
        </Section>
      ) : null}
      <Section header="Input">
        <Text selectable style={[type.mono, styles.code]}>
          {boundedPrettyPrint(part.input)}
        </Text>
      </Section>
      {part.output !== undefined ? (
        <Section header="Result">
          <Text selectable style={[type.mono, styles.code]}>
            {boundedPrettyPrint(part.output)}
          </Text>
        </Section>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.groupedBackground },
  content: { paddingTop: spacing.lg, paddingBottom: spacing.xxl },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
  },
  title: { flex: 1, color: colors.label },
  value: { color: colors.secondaryLabel, maxWidth: 200, textAlign: "right" },
  note: {
    color: colors.secondaryLabel,
    marginHorizontal: spacing.xl,
    marginTop: spacing.md,
  },
  summary: { color: colors.label, padding: spacing.lg },
  code: { color: colors.label, padding: spacing.lg, borderRadius: radius.md },
});
