import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { createStyles, spacing, type } from "@/lib/theme";

import { useResetExpert } from "../useResetExpert";
import { BulletList } from "./BulletList";

interface Props {
  expertID: string;
  name: string;
  /** Hired from the roster, so a reset also puts their soul and routines back. */
  hasTemplate: boolean;
  isOpen: boolean;
  onClose: () => void;
  onReset: () => void;
}

export function ResetExpertSheet({
  expertID,
  name,
  hasTemplate,
  isOpen,
  onClose,
  onReset,
}: Props) {
  const styles = useStyles();
  const reset = useResetExpert({ expertID, name, isOpen, onReset, onClose });

  // While the reset runs nothing closes the sheet: it cannot be undone, and
  // its outcome must be seen.
  function close() {
    if (!reset.isResetting && !reset.isOpeningChat) onClose();
  }

  function renderPhase() {
    if (reset.phase === "done") {
      return (
        <>
          <Text accessibilityRole="header" style={styles.heading}>
            {`${name} is reset`}
          </Text>
          {renderError()}
          <View style={styles.buttons}>
            <Button
              label={`Chat with ${name}`}
              onPress={reset.chat}
              isLoading={reset.isOpeningChat}
              fullWidth
            />
            {renderClose()}
          </View>
        </>
      );
    }
    if (reset.phase === "busy") {
      return (
        <>
          <Text accessibilityRole="header" style={styles.heading}>
            {`${name} is still working`}
          </Text>
          <Text style={styles.text}>Reset once they finish.</Text>
          {renderError()}
          <View style={styles.buttons}>
            <Button
              label="Try again"
              onPress={reset.confirm}
              isLoading={reset.isResetting}
              fullWidth
            />
            {renderClose()}
          </View>
        </>
      );
    }
    return (
      <>
        <Text accessibilityRole="header" style={styles.heading}>
          {`Reset ${name}?`}
        </Text>
        <ResetConfirm name={name} hasTemplate={hasTemplate} />
        {renderError()}
        <View style={styles.buttons}>
          <Button
            label="Reset"
            variant="destructive"
            onPress={reset.confirm}
            isLoading={reset.isResetting}
            fullWidth
          />
          <Button
            label="Cancel"
            variant="secondary"
            onPress={close}
            disabled={reset.isResetting}
            fullWidth
          />
        </View>
      </>
    );
  }

  function renderClose() {
    return (
      <Button
        label="Close"
        variant="secondary"
        onPress={close}
        disabled={reset.isResetting || reset.isOpeningChat}
        fullWidth
      />
    );
  }

  function renderError() {
    if (!reset.error) return null;
    return (
      <Text accessibilityLiveRegion="polite" style={styles.error}>
        {reset.error}
      </Text>
    );
  }

  return (
    <BottomSheet isOpen={isOpen} onClose={close}>
      <View style={styles.body}>{renderPhase()}</View>
    </BottomSheet>
  );
}

interface ConfirmProps {
  name: string;
  hasTemplate: boolean;
}

// Word for word the web's confirm step.
function ResetConfirm({ name, hasTemplate }: ConfirmProps) {
  const styles = useStyles();
  const intro = hasTemplate
    ? `${name} goes back to how they were the day you hired them.`
    : `${name} goes back to how they were the day you created them, keeping their soul and routines.`;

  return (
    <>
      <Text style={styles.text}>
        {`${intro} This can't be undone. If ${name} is working on something, wait for them to finish first.`}
      </Text>
      <BulletList
        label="Erased"
        lines={[
          `Everything ${name} remembers`,
          `${name}'s computer and the notes on it`,
          ...(hasTemplate
            ? [
                "Your edits to their soul and routines",
                `Routines you switched on are switched off. The ones ${name} came with come back, also off`,
              ]
            : []),
        ]}
      />
      <BulletList
        label="Removed"
        lines={[
          `Your chats with ${name}. They can't be opened again, even from an old link`,
          "Approvals from those chats waiting in Needs you",
        ]}
      />
      <BulletList
        label="Kept"
        lines={[
          `The skills ${name} has learned`,
          "Their workflows, and the schedules on them",
          "Their connections, budget and run history",
          "Your files, and what the rest of your team knows",
          ...(hasTemplate
            ? []
            : ["Their soul and routines, each routine still on or off as it is now"]),
        ]}
      />
    </>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  heading: { ...type.headline, color: colors.text },
  text: { ...type.body, color: colors.textSecondary },
  error: { ...type.small, color: colors.error },
  buttons: { gap: spacing.md, marginTop: spacing.sm },
}));
