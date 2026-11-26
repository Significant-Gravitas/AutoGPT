import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { setAnalyticsConsent } from "@/lib/analytics/client";
import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  onAnswer: () => void;
}

/** Asks once whether usage data may be sent. Closing it counts as "Not now". */
export function UsageDataSheet({ onAnswer }: Props) {
  const styles = useStyles();

  function answer(allowed: boolean) {
    setAnalyticsConsent(allowed);
    onAnswer();
  }

  return (
    <BottomSheet isOpen onClose={() => answer(false)}>
      <View style={styles.body}>
        <Text accessibilityRole="header" style={styles.heading}>
          Help improve AutoGPT
        </Text>
        <Text style={styles.text}>
          Share which screens you open and which buttons you tap. Never what you write or what
          your experts do.
        </Text>
        <View style={styles.buttons}>
          <Button label="Share usage data" onPress={() => answer(true)} fullWidth />
          <Button
            label="Not now"
            variant="secondary"
            onPress={() => answer(false)}
            fullWidth
          />
        </View>
      </View>
    </BottomSheet>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  heading: { ...type.headline, color: colors.text },
  text: { ...type.body, color: colors.textSecondary },
  buttons: { gap: spacing.md, marginTop: spacing.sm },
}));
