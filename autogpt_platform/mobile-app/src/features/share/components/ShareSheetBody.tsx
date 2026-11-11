import { Link01Icon, Share08Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Pressable, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { RetryNotice } from "@/features/files/components/RetryNotice";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { useShareSheet } from "../useShareSheet";
import { IncludeRunsOption } from "./IncludeRunsOption";

interface Props {
  sessionID: string;
}

// The web share dialog's own warning, so both apps promise the same.
const CONSENT =
  "Anyone with the link will see this conversation. Don't share if it contains secrets you pasted, personal details, or credentials you wouldn't want public. Sharing is live: new messages, workflow runs, and files added after you enable sharing become visible too. Stop sharing to revoke access.";

export function ShareSheetBody({ sessionID }: Props) {
  const sheet = useShareSheet(sessionID);
  const styles = useStyles();
  const colors = useColors();

  function renderControls() {
    if (sheet.isLoading) {
      return <ActivityIndicator color={colors.textTertiary} style={styles.spinner} />;
    }
    if (sheet.isLoadError) {
      return (
        <RetryNotice
          centered
          message="Couldn't load sharing. Try again."
          onRetry={sheet.retry}
        />
      );
    }
    if (!sheet.isShared || !sheet.url) {
      return (
        <Button
          label="Create link"
          onPress={sheet.createLink}
          isLoading={sheet.isCreating}
          fullWidth
        />
      );
    }
    return (
      <>
        <View style={styles.link}>
          <Icon icon={Link01Icon} size={18} color={colors.textTertiary} />
          <Text
            selectable
            numberOfLines={1}
            ellipsizeMode="middle"
            style={styles.url}
          >
            {sheet.url}
          </Text>
        </View>
        {/* No clipboard module is installed; the system sheet offers Copy. */}
        <Button
          label="Share link"
          onPress={sheet.sendLink}
          leadingIcon={Share08Icon}
          fullWidth
        />
        <Pressable
          accessibilityRole="button"
          accessibilityState={{ busy: sheet.isStopping }}
          disabled={sheet.isStopping}
          onPress={sheet.stopSharing}
          style={({ pressed }) => [styles.stop, pressed && styles.pressed]}
        >
          {sheet.isStopping ? (
            <ActivityIndicator size="small" color={colors.error} />
          ) : null}
          <Text style={styles.stopLabel}>Stop sharing</Text>
        </Pressable>
      </>
    );
  }

  return (
    <View style={styles.body}>
      <Text style={styles.consent}>{CONSENT}</Text>
      {sheet.isLoading || sheet.isLoadError ? null : (
        <IncludeRunsOption
          value={sheet.includeRuns}
          onChange={sheet.setIncludeRuns}
          disabled={sheet.isShared || sheet.isCreating}
        />
      )}
      {renderControls()}
      {sheet.error ? <Text style={styles.error}>{sheet.error}</Text> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  consent: { ...type.small, fontSize: 14, lineHeight: 20, color: colors.textSecondary },
  spinner: { paddingVertical: spacing.md },
  link: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    height: 46,
    borderRadius: radius.md,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.background,
  },
  url: { ...type.small, flex: 1, fontSize: 14, color: colors.text },
  // A ghost Button with its label in the error colour.
  stop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "stretch",
    gap: 8,
    height: 46,
    borderRadius: radius.pill,
  },
  pressed: { opacity: 0.85 },
  stopLabel: { fontSize: 15, lineHeight: 20, fontWeight: "500", color: colors.error },
  error: { ...type.small, color: colors.error },
}));
