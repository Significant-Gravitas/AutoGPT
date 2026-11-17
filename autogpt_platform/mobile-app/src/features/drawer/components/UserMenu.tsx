import {
  ArrowUp01Icon,
  Logout01Icon,
  Settings01Icon,
} from "@hugeicons/core-free-icons";
import { useRef, useState, type ComponentRef } from "react";
import {
  Modal,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource } from "@/features/team/helpers";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { DrawerRow } from "../DrawerRow";
import { AppearancePicker } from "./AppearancePicker";

const AVATAR_SIZE = 36;

interface Props {
  name: string | null;
  email: string | null;
  image: string | null;
  onOpenSettings: () => void;
  onSignOut: () => void;
}

interface Anchor {
  left: number;
  width: number;
  bottom: number;
}

/** The signed-in person's row; tapping it opens a small menu just above. */
export function UserMenu({ name, email, image, onOpenSettings, onSignOut }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const { height: windowHeight } = useWindowDimensions();
  const rowRef = useRef<ComponentRef<typeof View>>(null);
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  const title = name?.trim() || email || "Account";
  const detail = name?.trim() ? email : null;

  function open() {
    rowRef.current?.measureInWindow((x, y, width) => {
      setAnchor({ left: x, width, bottom: windowHeight - y + spacing.xs });
    });
  }

  function close() {
    setAnchor(null);
  }

  function handleSettings() {
    close();
    onOpenSettings();
  }

  function handleSignOut() {
    close();
    onSignOut();
  }

  return (
    <>
      <Pressable
        ref={rowRef}
        accessibilityRole="button"
        accessibilityLabel={`${title}, account menu`}
        accessibilityState={{ expanded: anchor !== null }}
        onPress={open}
        style={({ pressed }) => [styles.row, pressed && styles.pressed]}
      >
        <ExpertAvatar
          name={title}
          source={avatarSource({ avatar_url: image })}
          size={AVATAR_SIZE}
        />
        <View style={styles.text}>
          <Text numberOfLines={1} style={styles.title}>
            {title}
          </Text>
          {detail ? (
            <Text numberOfLines={1} style={styles.detail}>
              {detail}
            </Text>
          ) : null}
        </View>
        <Icon icon={ArrowUp01Icon} size={16} color={colors.textTertiary} />
      </Pressable>
      <Modal
        transparent
        visible={anchor !== null}
        animationType="fade"
        onRequestClose={close}
        statusBarTranslucent
      >
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Close menu"
          onPress={close}
          style={StyleSheet.absoluteFill}
        />
        {anchor ? (
          <View accessibilityRole="menu" style={[styles.menu, anchor]}>
            <AppearancePicker />
            <View style={styles.divider} />
            <DrawerRow
              icon={Settings01Icon}
              label="Settings"
              onPress={handleSettings}
            />
            <DrawerRow
              icon={Logout01Icon}
              label="Sign out"
              onPress={handleSignOut}
            />
          </View>
        ) : null}
      </Modal>
    </>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  text: { flex: 1, minWidth: 0 },
  title: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
  divider: {
    height: StyleSheet.hairlineWidth,
    marginVertical: spacing.xs,
    backgroundColor: colors.border,
  },
  menu: {
    position: "absolute",
    padding: spacing.xs,
    gap: 2,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.background,
    shadowColor: "#000000",
    shadowOpacity: 0.16,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 8,
  },
}));
