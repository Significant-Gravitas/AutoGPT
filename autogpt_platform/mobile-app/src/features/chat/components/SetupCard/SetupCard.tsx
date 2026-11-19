import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { AccountPickerSheet } from "@/features/connections/components/AccountPickerSheet";
import { CredentialSheet } from "@/features/connections/components/CredentialSheet/CredentialSheet";
import { createStyles, radius, spacing, type } from "@/lib/theme";

import { FollowUpSender } from "./components/FollowUpSender";
import { SetupRow } from "./components/SetupRow";
import { actionLabel, rowHint, type SetupRequest } from "./helpers";
import { useSetupCard } from "./useSetupCard";

interface Props {
  sessionID: string;
  toolCallID: string;
  setup: SetupRequest;
  /** Text of the message after the card, or null while the card is the last thing in the chat. */
  replyText: string | null;
  isBusy: boolean;
  onSend: (text: string) => boolean;
}

/** The services a tool needs before it can go on, each with its own way in. */
export function SetupCard(props: Props) {
  const styles = useStyles();
  const card = useSetupCard(props);
  const { run, connect } = card;

  const showContinue = card.canContinue && !card.shouldAutoSend;
  const hasFooter =
    run.followUp !== "idle" || !!run.error || card.needsWeb || connect.needsPlan || showContinue;

  function renderFooter() {
    if (run.followUp === "sent") {
      return <Text style={styles.footerText}>Connected. Continuing…</Text>;
    }
    if (run.followUp === "sending") {
      return <Text style={styles.footerText}>Continuing…</Text>;
    }
    return (
      <>
        {run.error ? <Text style={styles.error}>{run.error}</Text> : null}
        {props.setup.needsInputs ? (
          <Text style={styles.footerText}>
            {card.rows.length > 0
              ? "This also needs details you can fill in on the web."
              : "This needs details you can fill in on the web."}
          </Text>
        ) : null}
        {connect.needsPlan ? (
          <Text style={styles.footerText}>Connecting a service needs an active plan.</Text>
        ) : null}
        <View style={styles.actions}>
          {card.needsWeb ? (
            <Button
              label="Finish this on the web"
              onPress={connect.openOnWeb}
              variant="outline"
              size="small"
              leadingIcon={LinkSquare02Icon}
            />
          ) : null}
          {showContinue ? (
            <Button
              label={run.error ? "Try again" : "Continue"}
              onPress={card.handleContinue}
              size="small"
              disabled={props.isBusy}
            />
          ) : null}
        </View>
      </>
    );
  }

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text accessibilityRole="header" style={styles.title}>
          {card.rows.length > 0 ? "Connect to continue" : "More details needed"}
        </Text>
        {card.intro ? <Text style={styles.intro}>{card.intro}</Text> : null}
      </View>
      {card.rows.map(({ request, name, logoID, status, usesToken }, index) => (
        <SetupRow
          key={request.key}
          name={name}
          logoID={logoID}
          hint={rowHint(request, status, props.setup)}
          error={connect.errors[request.key] ?? null}
          isConnected={status.kind === "connected"}
          actionLabel={usesToken ? "Add token" : actionLabel(status)}
          isBusy={connect.busyKey === request.key}
          isLoading={card.isLoading && !run.picks[request.key]}
          isLocked={card.isAnyBusy || run.followUp !== "idle"}
          onPress={() => card.handleRow(index)}
        />
      ))}
      {hasFooter ? <View style={styles.footer}>{renderFooter()}</View> : null}
      {card.shouldAutoSend ? <FollowUpSender onReady={card.handleContinue} /> : null}
      <CredentialSheet
        target={connect.sheet}
        onClose={connect.closeSheet}
        onSaved={connect.handleSaved}
      />
      <AccountPickerSheet
        choice={connect.choice}
        onPick={connect.handlePick}
        onClose={connect.closeChoice}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    overflow: "hidden",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.surface,
  },
  header: { gap: 2, paddingHorizontal: spacing.md, paddingTop: spacing.md, paddingBottom: spacing.sm },
  title: { ...type.label, color: colors.text },
  intro: { ...type.small, color: colors.textSecondary },
  footer: {
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  footerText: { ...type.small, color: colors.textTertiary },
  error: { ...type.small, color: colors.error },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
}));
