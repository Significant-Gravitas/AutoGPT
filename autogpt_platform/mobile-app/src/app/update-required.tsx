import { Download04Icon } from "@hugeicons/core-free-icons";
import Constants from "expo-constants";
import { Image } from "expo-image";
import { Linking, Platform, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { storeURL } from "@/features/update/helpers";
import { APP_VERSION } from "@/features/update/useUpdateGate";
import { createStyles, spacing, type } from "@/lib/theme";

// The app's own icon, as on the sign-in screen.
const appIcon = require("../../assets/images/icon.png");

const STORE_URL = storeURL(
  Platform.OS,
  Constants.expoConfig?.android?.package ?? null,
  process.env.EXPO_PUBLIC_APP_STORE_ID || null,
);

/** A version the server no longer lets in stops here, signed in or not. */
export default function UpdateRequiredScreen() {
  const styles = useStyles();
  const insets = useSafeAreaInsets();
  const storeName = Platform.OS === "ios" ? "the App Store" : "Google Play";

  function openStore() {
    if (STORE_URL) void Linking.openURL(STORE_URL).catch(() => undefined);
  }

  return (
    <View style={[styles.screen, { paddingTop: insets.top, paddingBottom: insets.bottom + spacing.lg }]}>
      <View style={styles.middle}>
        <Image source={appIcon} style={styles.icon} accessibilityIgnoresInvertColors />
        <Text accessibilityRole="header" style={styles.heading}>
          Update AutoGPT
        </Text>
        <Text style={styles.body}>
          {`This version of the app no longer works. Get the latest one from ${storeName} to keep going.`}
        </Text>
        {APP_VERSION ? <Text style={styles.version}>{`You have version ${APP_VERSION}`}</Text> : null}
      </View>
      {STORE_URL ? (
        <Button label="Update" leadingIcon={Download04Icon} onPress={openStore} fullWidth />
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, paddingHorizontal: spacing.xl, backgroundColor: colors.background },
  middle: { flex: 1, alignItems: "center", justifyContent: "center", gap: spacing.md },
  icon: { width: 64, height: 64, borderRadius: 16, marginBottom: spacing.xs },
  heading: { ...type.headline, color: colors.text, textAlign: "center" },
  body: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  version: { ...type.small, color: colors.textTertiary },
}));
