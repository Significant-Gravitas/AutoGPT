import { Text } from "react-native";

import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  prompt: string;
  linkLabel: string;
  onPress: () => void;
}

/** "New here? Create an account": a line under an auth form with one link. */
export function AuthFooter({ prompt, linkLabel, onPress }: Props) {
  const styles = useStyles();
  return (
    <Text style={styles.footer}>
      {prompt}
      <Text accessibilityRole="link" onPress={onPress} style={styles.link}>
        {linkLabel}
      </Text>
    </Text>
  );
}

const useStyles = createStyles((colors) => ({
  footer: {
    ...type.small,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.lg,
  },
  link: {
    color: colors.text,
    fontWeight: "500",
    textDecorationLine: "underline",
  },
}));
