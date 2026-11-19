import { View } from "react-native";

import { Button } from "@/components/Button";
import { markOnboarded } from "@/features/onboarding/onboardedStore";
import { useSession } from "@/lib/auth/nativeSession";
import { createStyles } from "@/lib/theme";

export default function OnboardingScreen() {
  const session = useSession();
  const styles = useStyles();

  function finish() {
    if (session) markOnboarded(session.user.id);
  }

  return (
    <View style={styles.screen}>
      <Button label="Continue" onPress={finish} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
}));
