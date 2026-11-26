import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Text, View } from "react-native";

import { Button } from "@/components/Button";
import type { CredentialMeta } from "@/lib/api/integrations";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";
import { useMountEffect } from "@/lib/useMountEffect";

import { logoID } from "../../helpers";
import { BottomSheet } from "../BottomSheet";
import { ProviderTile } from "../ProviderTile";
import { useDeviceCodeSheet } from "./useDeviceCodeSheet";

/** The row a ChatGPT sign-in answers. */
export interface DeviceCodeTarget {
  rowKey: string;
  provider: string;
  providerName: string;
  /** An account to sign in again in place. */
  credentialID?: string;
}

interface Props {
  /** Null keeps it closed. */
  target: DeviceCodeTarget | null;
  onClose: () => void;
  onSaved: (rowKey: string, credential: CredentialMeta) => void;
  onPlanRequired: () => void;
}

/** Signs in to ChatGPT with a code entered on ChatGPT's page. */
export function DeviceCodeSheet({ target, ...props }: Props) {
  // Mounted only while open, so each opening starts a new sign-in.
  if (!target) return null;
  return <OpenSheet key={target.rowKey} target={target} {...props} />;
}

function OpenSheet({ target, onClose, onSaved, onPlanRequired }: Props & { target: DeviceCodeTarget }) {
  const sheet = useDeviceCodeSheet({
    credentialID: target.credentialID,
    onSaved: (credential) => onSaved(target.rowKey, credential),
    onClose,
    onPlanRequired,
  });
  return (
    <BottomSheet isOpen onClose={sheet.close}>
      <DeviceCodeView target={target} sheet={sheet} />
    </BottomSheet>
  );
}

/** The same steps inside a sheet that is already open, with a way back. */
export function DeviceCodeForm({ target, onClose, onSaved, onPlanRequired }: Props & { target: DeviceCodeTarget }) {
  const sheet = useDeviceCodeSheet({
    credentialID: target.credentialID,
    onSaved: (credential) => onSaved(target.rowKey, credential),
    onClose,
    onPlanRequired,
  });
  return (
    <>
      <DeviceCodeView target={target} sheet={sheet} />
      <Button label="Back" onPress={sheet.close} variant="ghost" size="small" />
    </>
  );
}

interface ViewProps {
  target: DeviceCodeTarget;
  sheet: ReturnType<typeof useDeviceCodeSheet>;
}

function DeviceCodeView({ target, sheet }: ViewProps) {
  const styles = useStyles();
  const colors = useColors();

  function renderBody() {
    if (sheet.cantStart) return <Text style={styles.error}>{sheet.cantStart}</Text>;
    if (sheet.failure) {
      return (
        <>
          <Text style={styles.error}>{sheet.failure}</Text>
          <Button label="Try again" onPress={sheet.retry} fullWidth />
        </>
      );
    }
    if (sheet.isStarting || !sheet.userCode) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    if (sheet.isFinishing || sheet.shouldFinish) {
      return (
        <>
          <Text style={styles.intro}>Connecting…</Text>
          {sheet.shouldFinish ? <Finisher onReady={sheet.finish} /> : null}
        </>
      );
    }
    return (
      <>
        <Text style={styles.intro}>Enter this code on ChatGPT&apos;s page, then come back here.</Text>
        <View style={styles.codeBox}>
          <Text style={styles.code} selectable accessibilityLabel={`Code ${sheet.userCode}`}>
            {sheet.userCode}
          </Text>
          <Button
            label={sheet.isCopied ? "Copied" : "Copy"}
            onPress={sheet.copyCode}
            variant="outline"
            size="small"
            accessibilityLabel="Copy the code"
          />
        </View>
        <Button label="Open ChatGPT" onPress={sheet.openChatGPT} leadingIcon={LinkSquare02Icon} fullWidth />
      </>
    );
  }

  return (
    <View style={styles.form}>
      <View style={styles.heading}>
        <ProviderTile name={target.providerName} logoID={logoID(target.provider)} />
        <Text accessibilityRole="header" style={styles.title} numberOfLines={1}>
          {target.providerName}
        </Text>
      </View>
      {renderBody()}
    </View>
  );
}

/** Mounted once the sign-in is done; saves the account once. */
function Finisher({ onReady }: { onReady: () => void }) {
  useMountEffect(() => {
    onReady();
  });
  return null;
}

const useStyles = createStyles((colors) => ({
  form: { gap: spacing.md },
  heading: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  title: { ...type.heading, color: colors.text, flex: 1 },
  intro: { ...type.small, fontSize: 14, lineHeight: 20, color: colors.textSecondary },
  error: { ...type.small, color: colors.error },
  loading: { paddingVertical: spacing.xl },
  codeBox: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.md,
    padding: spacing.lg,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  code: { ...type.title, flex: 1, letterSpacing: 2, color: colors.text },
}));
