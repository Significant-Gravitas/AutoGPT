import { Linking, Text } from "react-native";

import { PRIVACY_POLICY_URL, TERMS_OF_USE_URL } from "@/lib/legal";
import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  optedOut: boolean;
  onToggleOptOut: () => void;
  /** While the account is being created: the choice already went with it. */
  disabled?: boolean;
}

export function LegalLine({ optedOut, onToggleOptOut, disabled }: Props) {
  const styles = useStyles();

  function openTerms() {
    Linking.openURL(TERMS_OF_USE_URL).catch(() => undefined);
  }

  function openPrivacy() {
    Linking.openURL(PRIVACY_POLICY_URL).catch(() => undefined);
  }

  return (
    <Text style={styles.legal}>
      By continuing you agree to our{" "}
      <Text accessibilityRole="link" onPress={openTerms} style={styles.link}>
        Terms of Use
      </Text>{" "}
      and{" "}
      <Text accessibilityRole="link" onPress={openPrivacy} style={styles.link}>
        Privacy Policy
      </Text>
      .{" "}
      {optedOut ? (
        <>
          <Text style={styles.strong}>You won&apos;t get marketing emails.</Text>{" "}
          <Text
            accessibilityRole="button"
            accessibilityState={{ disabled: !!disabled }}
            onPress={disabled ? undefined : onToggleOptOut}
            style={styles.link}
          >
            Undo
          </Text>
          .
        </>
      ) : (
        <>
          We may email you product updates and offers;{" "}
          <Text
            accessibilityRole="button"
            accessibilityState={{ disabled: !!disabled }}
            onPress={disabled ? undefined : onToggleOptOut}
            style={styles.link}
          >
            opt out
          </Text>
          .
        </>
      )}
    </Text>
  );
}

const useStyles = createStyles((colors) => ({
  legal: {
    ...type.small,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.xs,
  },
  link: {
    color: colors.text,
    fontWeight: "500",
    textDecorationLine: "underline",
  },
  strong: { color: colors.text },
}));
