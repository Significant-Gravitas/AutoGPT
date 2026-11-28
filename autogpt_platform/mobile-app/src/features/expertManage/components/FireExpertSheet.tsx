import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { createStyles, spacing, type } from "@/lib/theme";

import { useFireExpert } from "../useFireExpert";
import { BulletList } from "./BulletList";

interface Props {
  expertID: string;
  name: string;
  isOpen: boolean;
  onClose: () => void;
}

export function FireExpertSheet({ expertID, name, isOpen, onClose }: Props) {
  const styles = useStyles();
  const fire = useFireExpert({ expertID, isOpen, onFired: onClose });

  // While the request runs nothing closes the sheet, or its outcome is lost.
  function close() {
    if (!fire.isFiring) onClose();
  }

  return (
    <BottomSheet isOpen={isOpen} onClose={close}>
      <View style={styles.body}>
        <Text accessibilityRole="header" style={styles.heading}>
          {`Fire ${name}?`}
        </Text>
        <BulletList
          lines={[
            "Installed workflows stay in your library.",
            fire.pauseLine,
            "Any chat history stays available but read-only.",
            "Their work stays yours.",
          ]}
        />
        {fire.error ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {fire.error}
          </Text>
        ) : null}
        <View style={styles.buttons}>
          <Button
            label={`Fire ${name}`}
            variant="destructive"
            onPress={fire.confirm}
            isLoading={fire.isFiring}
            disabled={fire.isPreviewLoading}
            fullWidth
          />
          <Button
            label={`Keep ${name}`}
            variant="secondary"
            onPress={close}
            disabled={fire.isFiring}
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
  error: { ...type.small, color: colors.error },
  buttons: { gap: spacing.md, marginTop: spacing.sm },
}));
