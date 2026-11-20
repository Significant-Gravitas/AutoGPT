import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { createStyles, spacing, type } from "@/lib/theme";

import { SheetField } from "../../CredentialSheet/components/SheetField";

interface Props {
  value: string;
  onChange: (text: string) => void;
  canSubmit: boolean;
  isBusy: boolean;
  error: string | null;
  onSubmit: () => void;
  onBack: () => void;
}

/** Asks for an MCP server's address before signing in to it. */
export function ServerStep({ value, onChange, canSubmit, isBusy, error, onSubmit, onBack }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.step}>
      <Text accessibilityRole="header" style={styles.title}>
        MCP server
      </Text>
      <Text style={styles.intro}>
        {"Paste the server's address. If it has a sign-in, it opens next; if not, you can add a token."}
      </Text>
      <SheetField
        label="Server URL"
        value={value}
        onChangeText={onChange}
        placeholder="https://mcp.example.com/mcp"
        autoFocus
        onSubmitEditing={onSubmit}
      />
      {error ? <Text style={styles.error}>{error}</Text> : null}
      <View style={styles.actions}>
        <Button label="Back" onPress={onBack} variant="ghost" size="small" disabled={isBusy} />
        <Button
          label={isBusy ? "Signing in…" : "Continue"}
          onPress={onSubmit}
          size="small"
          disabled={!canSubmit}
          isLoading={isBusy}
        />
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  step: { gap: spacing.md },
  title: { ...type.heading, color: colors.text },
  intro: { ...type.small, fontSize: 14, lineHeight: 20, color: colors.textSecondary },
  error: { ...type.small, color: colors.error },
  actions: { flexDirection: "row", justifyContent: "space-between" },
}));
