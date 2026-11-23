import { useLocalSearchParams } from "expo-router";
import { ActivityIndicator, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { useOAuthReturn } from "@/features/connections/useOAuthReturn";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

/** Where `OAUTH_RETURN_URL` lands when it opens the app. */
export default function OAuthReturnScreen() {
  const params = useLocalSearchParams();
  const { error, goBack, canRetry, isRetrying, retryGrant } = useOAuthReturn(params);
  const styles = useStyles();
  const colors = useColors();

  return (
    <View style={styles.screen}>
      {error ? (
        <>
          <Text style={styles.text}>{error}</Text>
          {canRetry ? (
            <Button
              label="Try again"
              onPress={() => void retryGrant()}
              isLoading={isRetrying}
              size="small"
            />
          ) : null}
          <Button label="Back to the chat" onPress={goBack} variant="secondary" size="small" />
        </>
      ) : (
        <>
          <ActivityIndicator color={colors.textTertiary} />
          <Text style={styles.text}>Finishing sign-in…</Text>
        </>
      )}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.md,
    padding: spacing.xl,
    backgroundColor: colors.background,
  },
  text: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
