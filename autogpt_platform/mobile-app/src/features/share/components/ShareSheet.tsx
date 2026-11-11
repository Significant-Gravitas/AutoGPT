import { Modal, Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing } from "@/lib/theme";

import { ShareSheetBody } from "./ShareSheetBody";

interface Props {
  sessionID: string;
  isOpen: boolean;
  onClose: () => void;
}

export function ShareSheet({ sessionID, isOpen, onClose }: Props) {
  const insets = useSafeAreaInsets();
  const styles = useStyles();

  return (
    <Modal
      visible={isOpen}
      transparent
      animationType="fade"
      statusBarTranslucent
      onRequestClose={onClose}
    >
      <View style={styles.screen}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Close"
          onPress={onClose}
          style={styles.backdrop}
        />
        <View style={[styles.sheet, { paddingBottom: insets.bottom + spacing.lg }]}>
          <View style={styles.handle} />
          <Text accessibilityRole="header" style={styles.title}>
            Share this chat
          </Text>
          <ShareSheetBody sessionID={sessionID} />
        </View>
      </View>
    </Modal>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, justifyContent: "flex-end" },
  backdrop: { ...StyleSheet.absoluteFill, backgroundColor: "rgba(0,0,0,0.5)" },
  sheet: {
    gap: spacing.md,
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.sm,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    backgroundColor: colors.surface,
  },
  handle: {
    alignSelf: "center",
    width: 36,
    height: 4,
    borderRadius: 2,
    marginBottom: spacing.sm,
    backgroundColor: colors.borderStrong,
  },
  title: { fontSize: 20, lineHeight: 26, fontWeight: "600", color: colors.text },
}));
