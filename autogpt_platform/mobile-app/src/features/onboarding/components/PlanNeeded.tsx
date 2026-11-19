import { Image } from "expo-image";
import { Linking, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { FRONTEND_URL } from "@/lib/env";
import { OTTO_AVATAR_URL } from "@/lib/otto";
import { createStyles, spacing, type } from "@/lib/theme";

import { markOnboarded } from "../onboardedStore";

interface Props {
  userID: string;
}

/** The end on a server that wants a plan first: the web takes it from here. */
export function PlanNeeded({ userID }: Props) {
  const styles = useStyles();

  function openWeb() {
    Linking.openURL(FRONTEND_URL).catch(() => undefined);
    markOnboarded(userID);
  }

  function goHome() {
    markOnboarded(userID);
  }

  return (
    <View style={styles.screen}>
      <Image
        source={OTTO_AVATAR_URL}
        style={styles.otto}
        contentFit="contain"
        accessibilityLabel="Otto"
      />
      <Text accessibilityRole="header" style={styles.heading}>
        One step left: choose a plan
      </Text>
      <Text style={styles.body}>
        Your answers are saved. Choose a plan on the web, then come back and
        your team gets to work.
      </Text>
      <View style={styles.buttons}>
        <Button label="Open AutoGPT on the web" onPress={openWeb} fullWidth />
        <Button label="Go to Home" variant="secondary" onPress={goHome} fullWidth />
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.lg,
    paddingHorizontal: spacing.xl,
    backgroundColor: colors.background,
  },
  otto: { width: 104, height: 104 },
  heading: { ...type.headline, color: colors.text, textAlign: "center" },
  body: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  buttons: { alignSelf: "stretch", gap: spacing.md, marginTop: spacing.lg },
}));
