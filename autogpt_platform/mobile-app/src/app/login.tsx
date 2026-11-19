import { Image } from "expo-image";
import { router } from "expo-router";
import { Text, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { Button } from "@/components/Button";
import { AuthField } from "@/features/auth/components/AuthField";
import { AuthFooter } from "@/features/auth/components/AuthFooter";
import { useLoginScreen } from "@/features/auth/useLoginScreen";
import { createStyles, spacing, type } from "@/lib/theme";

const appIcon = require("../../assets/images/icon.png");

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
  const styles = useStyles();

  function goToSignup() {
    router.push("/signup");
  }

  return (
    <KeyboardAvoidingView
      behavior="padding"
      style={styles.screen}
    >
      <View style={styles.column}>
        <Image
          source={appIcon}
          style={styles.icon}
          accessibilityIgnoresInvertColors
        />
        <Text style={styles.title}>Sign in to AutoGPT</Text>
        <AuthField
          accessibilityLabel="Email"
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
          autoCorrect={false}
          autoComplete="email"
          textContentType="username"
          editable={!isSubmitting}
        />
        <AuthField
          accessibilityLabel="Password"
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          autoComplete="current-password"
          textContentType="password"
          editable={!isSubmitting}
        />
        {error ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {error}
          </Text>
        ) : null}
        <Button
          label={isSubmitting ? "Signing in…" : "Sign in"}
          onPress={submit}
          isLoading={isSubmitting}
          disabled={!canSubmit && !isSubmitting}
          fullWidth
        />
        <AuthFooter
          prompt="New here? "
          linkLabel="Create an account"
          onPress={goToSignup}
        />
      </View>
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
  error: {
    ...type.small,
    alignSelf: "stretch",
    color: colors.error,
  },
}));
