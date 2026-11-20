import { Text, TextInput, View } from "react-native";

import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  label: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  isSecret?: boolean;
  autoFocus?: boolean;
  onSubmitEditing?: () => void;
}

/** One labelled field. Secrets never autocorrect, suggest or capitalise. */
export function SheetField({
  label,
  value,
  onChangeText,
  placeholder,
  isSecret = false,
  autoFocus = false,
  onSubmitEditing,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor={colors.textTertiary}
        accessibilityLabel={label}
        secureTextEntry={isSecret}
        autoCorrect={false}
        autoCapitalize="none"
        autoComplete="off"
        spellCheck={false}
        textContentType={isSecret ? "password" : "none"}
        importantForAutofill={isSecret ? "yes" : "no"}
        autoFocus={autoFocus}
        returnKeyType="done"
        onSubmitEditing={onSubmitEditing}
        style={styles.input}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  field: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xs,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  label: { fontSize: 12, lineHeight: 16, color: colors.textTertiary },
  input: { ...type.body, color: colors.text, paddingVertical: spacing.xs },
}));
