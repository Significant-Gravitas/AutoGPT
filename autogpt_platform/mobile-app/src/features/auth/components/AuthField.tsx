import { StyleSheet, TextInput, type TextInputProps } from "react-native";

import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

type Props = Omit<
  TextInputProps,
  "style" | "placeholderTextColor" | "selectionColor"
>;

/** The email and password fields of the sign-in and sign-up screens. */
export function AuthField(props: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <TextInput
      placeholderTextColor={colors.textTertiary}
      selectionColor={colors.brand}
      {...props}
      style={styles.input}
    />
  );
}

const useStyles = createStyles((colors) => ({
  input: {
    ...type.body,
    alignSelf: "stretch",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    color: colors.text,
  },
}));
