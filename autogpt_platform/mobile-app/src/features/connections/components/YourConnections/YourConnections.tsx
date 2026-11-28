import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { RetryNotice } from "@/features/files/components/RetryNotice";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { AddConnectionSheet } from "../AddConnectionSheet/AddConnectionSheet";
import { ConnectionRow } from "../ConnectionRow";
import { ServiceLogos } from "./components/ServiceLogos";
import { useYourConnections } from "./useYourConnections";

interface Props {
  sessionID: string;
  isOpen: boolean;
}

/**
 * Every account you have connected, under the files of a chat with Otto.
 * Removing one is a Settings job, so the rows have no button.
 */
export function YourConnections({ sessionID, isOpen }: Props) {
  const section = useYourConnections(isOpen);
  const styles = useStyles();
  const colors = useColors();

  function renderBody() {
    if (section.hasFailed) {
      return <RetryNotice message="Couldn't load your connections." onRetry={section.retry} />;
    }
    if (section.isLoading) {
      return (
        <View accessibilityLabel="Loading your connections" style={[styles.card, styles.placeholder]}>
          <View style={styles.placeholderLine} />
        </View>
      );
    }
    if (section.accounts.length === 0) {
      return (
        <View style={[styles.card, styles.empty]}>
          <ServiceLogos />
          <Text style={styles.emptyTitle}>Nothing connected yet</Text>
          <Text style={styles.emptyText}>
            Connect an account and Otto can read and act on it for you. Nothing is shared until you do.
          </Text>
          <View style={styles.emptyButton}>
            <Button
              label="Add connection"
              onPress={section.openAdd}
              leadingIcon={PlusSignIcon}
              fullWidth
            />
          </View>
        </View>
      );
    }
    return (
      <View style={styles.card}>
        {section.accounts.map((account, index) => (
          <View key={account.id} style={index > 0 && styles.divider}>
            <ConnectionRow
              name={account.name}
              detail={account.detail}
              logoID={account.logoID}
              status="Connected"
            />
          </View>
        ))}
        <Pressable
          accessibilityRole="button"
          onPress={section.openAdd}
          style={({ pressed }) => [styles.addRow, pressed && styles.pressed]}
        >
          <View style={styles.addTile}>
            <Icon icon={PlusSignIcon} size={16} color={colors.textSecondary} />
          </View>
          <Text style={styles.addLabel}>Add connection</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View>
      <View style={styles.heading}>
        <Text accessibilityRole="header" style={styles.title}>
          Connections
        </Text>
        <Text style={styles.subtitle}>Accounts Otto can use in this chat.</Text>
      </View>
      {renderBody()}
      <AddConnectionSheet isOpen={section.isAdding} onClose={section.closeAdd} sessionID={sessionID} expertID={null} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  heading: { paddingHorizontal: spacing.xs, paddingBottom: 10, gap: 2 },
  title: { fontSize: 20, lineHeight: 25, fontWeight: "700", letterSpacing: -0.2, color: colors.text },
  subtitle: { fontSize: 14, lineHeight: 19, color: colors.textSecondary },
  card: { borderRadius: radius.lg, backgroundColor: colors.panelCard, overflow: "hidden" },
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  placeholder: { paddingHorizontal: spacing.md, paddingVertical: spacing.lg },
  placeholderLine: { height: 14, width: "60%", borderRadius: 7, backgroundColor: colors.surfacePressed },
  empty: { alignItems: "center", paddingHorizontal: 18, paddingTop: 22, paddingBottom: 18 },
  emptyTitle: { marginTop: 14, fontSize: 16, lineHeight: 21, fontWeight: "600", color: colors.text },
  emptyText: {
    marginTop: 4,
    maxWidth: 270,
    fontSize: 14,
    lineHeight: 19,
    color: colors.textSecondary,
    textAlign: "center",
  },
  emptyButton: { alignSelf: "stretch", marginTop: spacing.lg },
  addRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    minHeight: 56,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.panelDivider,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  addTile: {
    width: 32,
    height: 32,
    borderRadius: 8,
    borderWidth: 1.5,
    borderStyle: "dashed",
    borderColor: colors.borderStrong,
    alignItems: "center",
    justifyContent: "center",
  },
  addLabel: { fontSize: 15, lineHeight: 20, fontWeight: "600", color: colors.text },
}));
