import { Mail01Icon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { resendLabel } from "../signupHelpers";
import { AuthFooter } from "./AuthFooter";

interface Props {
  email: string;
  error: string | null;
  cooldown: number;
  isChecking: boolean;
  isResending: boolean;
  onConfirm: () => void;
  onResend: () => void;
  onStartAgain: () => void;
}

export function CheckInbox({
  email,
  error,
  cooldown,
  isChecking,
  isResending,
  onConfirm,
  onResend,
  onStartAgain,
}: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <View style={styles.column}>
      <View style={styles.roundIcon}>
        <Icon icon={Mail01Icon} size={28} color={colors.accentText} />
      </View>
      <Text accessibilityRole="header" style={styles.title}>
        Check your inbox
      </Text>
      <Text style={styles.body}>
        We sent an email to <Text style={styles.email}>{email}</Text>. Open the
        link in it, then come back here.
      </Text>
      <Text style={styles.hint}>Can&apos;t find it? Check your spam folder.</Text>
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      <Button
        label="I've opened the link"
        onPress={onConfirm}
        isLoading={isChecking}
        fullWidth
      />
      <Button
        label={resendLabel(cooldown)}
        variant="secondary"
        onPress={onResend}
        isLoading={isResending}
        disabled={cooldown > 0}
        fullWidth
      />
      <AuthFooter
        prompt="Wrong address? "
        linkLabel="Start again"
        onPress={onStartAgain}
        // A check or resend still running belongs to this address.
        disabled={isChecking || isResending}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  column: {
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  roundIcon: {
    width: 56,
    height: 56,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.brandSoft,
    marginBottom: spacing.sm,
  },
  title: { ...type.title, color: colors.text, textAlign: "center" },
  body: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  email: { color: colors.text, fontWeight: "500" },
  hint: {
    ...type.small,
    color: colors.textSecondary,
    textAlign: "center",
    marginBottom: spacing.md,
  },
  error: { ...type.small, alignSelf: "stretch", color: colors.error },
}));
