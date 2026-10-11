import { router } from "expo-router";
import { useState } from "react";
import { StyleSheet, Text } from "react-native";
import { KeyboardAwareScrollView } from "react-native-keyboard-controller";

import { PrimaryButton } from "../../design/components/Buttons";
import { FieldRow, Section } from "../../design/components/Grouped";
import { colors, spacing, type } from "../../design/theme";
import { useSession } from "../../features/auth/SessionProvider";
import {
  createCustomServer,
  normalizeOrigin,
} from "../../features/servers/serverConfig";

export default function OwnServerScreen() {
  const { rememberServer } = useSession();
  const [name, setName] = useState("");
  const [authInput, setAuthInput] = useState("");
  const [apiInput, setAPIInput] = useState("");
  const [errors, setErrors] = useState<{
    auth: string | null;
    api: string | null;
  }>({
    auth: null,
    api: null,
  });
  const [busy, setBusy] = useState(false);

  async function submit() {
    const auth = normalizeOrigin(authInput);
    const api = normalizeOrigin(apiInput);
    setErrors({
      auth: auth.ok ? null : auth.reason,
      api: api.ok ? null : api.reason,
    });
    if (!auth.ok || !api.ok) return;
    setBusy(true);
    try {
      const server = await createCustomServer({
        name,
        authUrl: auth.origin,
        apiUrl: api.origin,
      });
      await rememberServer(server);
      router.push("/sign-in");
    } finally {
      setBusy(false);
    }
  }

  return (
    <KeyboardAwareScrollView
      bottomOffset={spacing.xl}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps="handled"
      style={styles.screen}
      contentContainerStyle={styles.content}
    >
      <Section
        header="Server"
        footer="Enter your AutoGPT web address, which handles sign-in, and its API address. Plain http:// only works for localhost."
      >
        <FieldRow
          label="Sign-in address"
          value={authInput}
          onChangeText={setAuthInput}
          placeholder="https://autogpt.example.com"
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          textContentType="URL"
          returnKeyType="next"
          testID="own-server-auth-url"
        />
        <FieldRow
          label="API address"
          value={apiInput}
          onChangeText={setAPIInput}
          placeholder="https://api.autogpt.example.com"
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          textContentType="URL"
          returnKeyType="next"
          testID="own-server-api-url"
        />
        <FieldRow
          label="Name (optional)"
          value={name}
          onChangeText={setName}
          placeholder="Team server"
          returnKeyType="done"
          onSubmitEditing={() => void submit()}
        />
      </Section>
      {errors.auth ? (
        <Text style={[type.footnote, styles.error]} accessibilityRole="alert">
          Sign-in address: {errors.auth}
        </Text>
      ) : null}
      {errors.api ? (
        <Text style={[type.footnote, styles.error]} accessibilityRole="alert">
          API address: {errors.api}
        </Text>
      ) : null}
      <Text style={[type.footnote, styles.note]}>
        Your server&apos;s addresses are checked when you sign in. Your sign-in
        is only ever sent to the sign-in address.
      </Text>
      <PrimaryButton
        title="Continue"
        onPress={() => void submit()}
        busy={busy}
        style={styles.button}
      />
    </KeyboardAwareScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.groupedBackground },
  content: { paddingBottom: spacing.xxl },
  error: {
    color: colors.danger,
    marginHorizontal: spacing.xl,
    marginTop: spacing.sm,
  },
  note: {
    color: colors.secondaryLabel,
    marginHorizontal: spacing.xl,
    marginTop: spacing.lg,
  },
  button: { marginHorizontal: spacing.lg, marginTop: spacing.xl },
});
