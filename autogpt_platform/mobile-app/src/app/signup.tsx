import { Image } from "expo-image";
import { router } from "expo-router";
import { Text, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { Button } from "@/components/Button";
import { AuthField } from "@/features/auth/components/AuthField";
import { AuthFooter } from "@/features/auth/components/AuthFooter";
import { CheckInbox } from "@/features/auth/components/CheckInbox";
import { LegalLine } from "@/features/auth/components/LegalLine";
import { useSignupScreen } from "@/features/auth/useSignupScreen";
import { createStyles, spacing, type } from "@/lib/theme";

const appIcon = require("../../assets/images/icon.png");

export default function SignupScreen() {
  const screen = useSignupScreen();
  const styles = useStyles();

  function goToSignIn() {
    if (router.canGoBack()) router.back();
    else router.replace("/login");
  }

  return (
    <KeyboardAvoidingView behavior="padding" style={styles.screen}>
      {screen.pendingEmail ? (
        <CheckInbox
          email={screen.pendingEmail}
          error={screen.error}
          cooldown={screen.cooldown}
          isChecking={screen.isChecking}
          isResending={screen.isResending}
          onConfirm={screen.confirmOpened}
          onResend={screen.resend}
          onStartAgain={screen.startAgain}
        />
      ) : (
        <View style={styles.column}>
          <Image
            source={appIcon}
            style={styles.icon}
            accessibilityIgnoresInvertColors
          />
          <Text accessibilityRole="header" style={styles.title}>
            Create your account
          </Text>
          <AuthField
            accessibilityLabel="Email"
            placeholder="Email"
            value={screen.email}
            onChangeText={screen.setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
            autoComplete="email"
            textContentType="username"
            editable={!screen.isSubmitting}
          />
          <AuthField
            accessibilityLabel="Password"
            placeholder="Password"
            value={screen.password}
            onChangeText={screen.setPassword}
            secureTextEntry
            autoComplete="new-password"
            textContentType="newPassword"
            editable={!screen.isSubmitting}
          />
          <Text style={styles.hint}>At least 12 characters.</Text>
          {screen.error ? (
            <Text accessibilityLiveRegion="polite" style={styles.error}>
              {screen.error}
            </Text>
          ) : null}
          <Button
            label={screen.isSubmitting ? "Creating account…" : "Create account"}
            onPress={screen.submit}
            isLoading={screen.isSubmitting}
            disabled={!screen.canSubmit && !screen.isSubmitting}
            fullWidth
          />
          <LegalLine
            optedOut={screen.optedOut}
            onToggleOptOut={screen.toggleOptOut}
          />
          <AuthFooter
            prompt="Already have an account? "
            linkLabel="Sign in"
            onPress={goToSignIn}
          />
        </View>
      )}
    </KeyboardAvoidingView>
  );
}

const useStyles = createStyles((colors) => ({
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
  hint: {
    ...type.small,
    alignSelf: "stretch",
    color: colors.textSecondary,
    marginTop: -spacing.xs,
  },
  error: {
    ...type.small,
    alignSelf: "stretch",
    color: colors.error,
  },
}));
