import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { createStyles, spacing, type } from "@/lib/theme";

import { useDeleteAccountSheet } from "./useDeleteAccountSheet";

interface Props {
  email: string;
  isOpen: boolean;
  onClose: () => void;
}

export function DeleteAccountSheet({ email, isOpen, onClose }: Props) {
  const styles = useStyles();
  const { isDeleting, error, deleteAccount } = useDeleteAccountSheet(email);

  function close() {
    if (!isDeleting) onClose();
  }

  return (
    <BottomSheet isOpen={isOpen} onClose={close}>
      <View style={styles.body}>
        <Text accessibilityRole="header" style={styles.heading}>
          Delete your account?
        </Text>
        <Text style={styles.text}>
          This closes your AutoGPT account now and erases it after 14 days. Your
          plan ends and your connected apps are disconnected. To keep it, sign
          in on the web before then and choose Keep my account.
        </Text>
        {error ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {error}
          </Text>
        ) : null}
        <View style={styles.buttons}>
          <Button
            label="Delete account"
            variant="destructive"
            onPress={deleteAccount}
            isLoading={isDeleting}
            fullWidth
          />
          <Button
            label="Keep my account"
            variant="secondary"
            onPress={close}
            disabled={isDeleting}
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
  error: { ...type.small, color: colors.error },
  buttons: { gap: spacing.md, marginTop: spacing.sm },
}));
