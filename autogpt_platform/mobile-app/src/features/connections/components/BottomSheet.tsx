import type { ReactNode } from "react";
import { Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing } from "@/lib/theme";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
}

/** The app's hand-made sheet (as in Share), lifted above the keyboard. */
export function BottomSheet({ isOpen, onClose, children }: Props) {
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
      <KeyboardAvoidingView behavior="padding" style={styles.screen}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Close"
          onPress={onClose}
          style={styles.backdrop}
        />
        <View style={[styles.sheet, { paddingBottom: insets.bottom + spacing.lg }]}>
          <View style={styles.handle} />
          {/* A long list scrolls instead of running off the screen. */}
          <ScrollView
            style={styles.body}
            contentContainerStyle={styles.content}
            keyboardShouldPersistTaps="handled"
            bounces={false}
          >
            {children}
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, justifyContent: "flex-end" },
  backdrop: { ...StyleSheet.absoluteFill, backgroundColor: "rgba(0,0,0,0.5)" },
  sheet: {
    maxHeight: "90%",
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.sm,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    backgroundColor: colors.surface,
  },
  body: { flexGrow: 0 },
  content: { gap: spacing.md },
  handle: {
    alignSelf: "center",
    width: 36,
    height: 4,
    borderRadius: 2,
    marginBottom: spacing.sm,
    backgroundColor: colors.borderStrong,
  },
}));
