import * as Haptics from "expo-haptics";
import { router } from "expo-router";
import { useRef, useState } from "react";
import { StyleSheet, Text, View, type TextInput } from "react-native";
import { KeyboardAwareScrollView } from "react-native-keyboard-controller";

import { PlainButton, PrimaryButton } from "../../design/components/Buttons";
import { FieldRow, Row, Section } from "../../design/components/Grouped";
import { Icon } from "../../design/components/Icon";
import { EmptyState } from "../../design/components/StateViews";
import { colors, spacing, type } from "../../design/theme";
import { SignInError } from "../../features/auth/authApi";
import { useSession } from "../../features/auth/SessionProvider";
import { hostLabel } from "../../features/servers/serverConfig";

export default function SignInScreen() {
  const { state, signIn } = useSession();
  const server = state.status === "signedOut" ? state.server : null;
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const passwordField = useRef<TextInput>(null);

  async function submit() {
    if (busy) return;
    if (!email.trim() || !password) {
      setError("Enter your email and password.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await signIn(email, password);
    } catch (failure) {
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      setError(
        failure instanceof SignInError
          ? failure.message
          : "Couldn't sign in. Try again.",
      );
      if (failure instanceof SignInError && failure.failure === "credentials")
        setPassword("");
    } finally {
      setBusy(false);
    }
  }

  if (!server) {
    return (
      <EmptyState
        symbol="server.rack"
        title="Choose a server"
        message="Pick AutoGPT Cloud or your own server before signing in."
        action={
          <PlainButton title="Choose Server" onPress={() => router.back()} />
        }
      />
    );
  }

  return (
    <KeyboardAwareScrollView
      bottomOffset={spacing.xl}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps="handled"
      style={styles.screen}
      contentContainerStyle={styles.content}
    >
      <Section header="Server">
        <Row
          title={server.name}
          subtitle={hostLabel(server.authUrl)}
          leading={
            <Icon
              name={server.kind === "cloud" ? "cloud" : "server.rack"}
              size={22}
              color={colors.tint}
            />
          }
          trailing={
            <PlainButton title="Change" onPress={() => router.back()} />
          }
        />
      </Section>
      <Section header="Account">
        <FieldRow
          label="Email"
          value={email}
          onChangeText={setEmail}
          placeholder="you@example.com"
          autoCapitalize="none"
          autoCorrect={false}
          autoComplete="email"
          keyboardType="email-address"
          textContentType="username"
          returnKeyType="next"
          onSubmitEditing={() => passwordField.current?.focus()}
          editable={!busy}
          testID="sign-in-email"
        />
        <FieldRow
          ref={passwordField}
          label="Password"
          value={password}
          onChangeText={setPassword}
          placeholder="Required"
          secureTextEntry
          autoComplete="current-password"
          textContentType="password"
          returnKeyType="go"
          onSubmitEditing={() => void submit()}
          editable={!busy}
          testID="sign-in-password"
        />
      </Section>
      {error ? (
        <Text
          style={[type.footnote, styles.error]}
          accessibilityRole="alert"
          testID="sign-in-error"
        >
          {error}
        </Text>
      ) : null}
      <PrimaryButton
        title="Sign In"
        onPress={() => void submit()}
        busy={busy}
        style={styles.button}
        testID="sign-in-submit"
      />
      <View style={styles.notes}>
        <Text style={[type.footnote, styles.note]}>
          Signing in with Google isn&apos;t available in the app yet. If your
          account only uses Google, sign in on the web for now.
        </Text>
        <Text style={[type.footnote, styles.note]}>
          New to AutoGPT? Create your account on the web, then sign in here.
        </Text>
      </View>
    </KeyboardAwareScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.groupedBackground },
  content: { paddingBottom: spacing.xxl },
  error: {
    color: colors.danger,
    marginHorizontal: spacing.xl,
    marginTop: spacing.md,
  },
  button: { marginHorizontal: spacing.lg, marginTop: spacing.xl },
  notes: {
    marginHorizontal: spacing.xl,
    marginTop: spacing.xl,
    gap: spacing.sm,
  },
  note: { color: colors.secondaryLabel },
});
