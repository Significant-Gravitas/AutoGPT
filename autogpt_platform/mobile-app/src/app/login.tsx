import { Image } from "expo-image";
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { useLoginScreen } from "@/features/auth/useLoginScreen";
import { colors, radius, spacing, type } from "@/lib/theme";

const appIcon = require("@/assets/images/icon.png");

export default function LoginScreen() {
  const {
    email,
    setEmail,
    password,
    setPassword,
    error,
    isSubmitting,
    canSubmit,
    submit,
  } = useLoginScreen();

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.screen}
    >
      <View style={styles.column}>
        <Image
          source={appIcon}
          style={styles.icon}
          accessibilityIgnoresInvertColors
        />
        <Text style={styles.title}>Sign in to AutoGPT</Text>
        <TextInput
          accessibilityLabel="Email"
          placeholder="Email"
          placeholderTextColor={colors.textTertiary}
          selectionColor={colors.brand}
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
          autoCorrect={false}
          autoComplete="email"
          textContentType="username"
          editable={!isSubmitting}
          style={styles.input}
        />
        <TextInput
          accessibilityLabel="Password"
          placeholder="Password"
          placeholderTextColor={colors.textTertiary}
          selectionColor={colors.brand}
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          autoComplete="current-password"
          textContentType="password"
          editable={!isSubmitting}
          style={styles.input}
        />
        {error ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {error}
          </Text>
        ) : null}
        <Pressable
          accessibilityRole="button"
          accessibilityState={{ disabled: !canSubmit, busy: isSubmitting }}
          disabled={!canSubmit}
          onPress={submit}
          style={({ pressed }) => [
            styles.button,
            pressed && styles.pressed,
            !canSubmit && styles.disabled,
          ]}
        >
          <Text style={styles.buttonText}>
            {isSubmitting ? "Signing in…" : "Sign in"}
          </Text>
        </Pressable>
        <Text style={styles.footer}>
          New here? Create your account at platform.agpt.co first.
        </Text>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    justifyContent: "center",
    backgroundColor: colors.background,
  },
  column: {
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  icon: {
    width: 64,
    height: 64,
    borderRadius: 16,
    marginBottom: spacing.sm,
  },
  title: {
    ...type.title,
    color: colors.text,
    textAlign: "center",
    marginBottom: spacing.md,
  },
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
  error: {
    ...type.small,
    alignSelf: "stretch",
    color: colors.error,
  },
  button: {
    alignSelf: "stretch",
    alignItems: "center",
    paddingVertical: spacing.md + 2,
    marginTop: spacing.sm,
    borderRadius: radius.md,
    backgroundColor: colors.brand,
  },
  pressed: { opacity: 0.8 },
  disabled: { opacity: 0.4 },
  buttonText: { ...type.label, color: colors.onBrand },
  footer: {
    ...type.small,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.lg,
  },
});
