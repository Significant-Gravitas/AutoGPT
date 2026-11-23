import { Logout01Icon, Settings01Icon } from "@hugeicons/core-free-icons";
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
import { avatarSource } from "@/features/team/helpers";
import { createStyles, radius, shadows, spacing, type, useScheme } from "@/lib/theme";

import { DrawerRow } from "../DrawerRow";

const BUTTON_SIZE = 48;
const AVATAR_SIZE = 38;
const MENU_WIDTH = 264;

interface Props {
  name: string | null;
  email: string | null;
  image: string | null;
  onOpenSettings: () => void;
  onSignOut: () => void;
}

interface Anchor {
  left: number;
  bottom: number;
}

/** The signed-in person's round avatar; tapping it opens a small menu just above. */
export function UserMenu({ name, email, image, onOpenSettings, onSignOut }: Props) {
  const styles = useStyles();
  const isLight = useScheme() === "light";
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const buttonRef = useRef<ComponentRef<typeof View>>(null);
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  const title = name?.trim() || email || "Account";
  const detail = name?.trim() ? email : null;

  function open() {
    // Right edges line up, so the menu grows left from the avatar, but never
    // past the window's left edge on a narrow phone.
    buttonRef.current?.measureInWindow((x, y, width) => {
      setAnchor({
        left: Math.max(
          Math.min(x + width, windowWidth - spacing.sm) - MENU_WIDTH,
          spacing.sm,
        ),
        bottom: windowHeight - y + spacing.sm,
      });
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
        ref={buttonRef}
        accessibilityRole="button"
        accessibilityLabel={`${title}, account menu`}
        accessibilityState={{ expanded: anchor !== null }}
        onPress={open}
        style={({ pressed }) => [
          styles.button,
          isLight && styles.buttonShadow,
          pressed && styles.pressed,
        ]}
      >
        <ExpertAvatar
          name={title}
          source={avatarSource({ avatar_url: image })}
          size={AVATAR_SIZE}
        />
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
            <View style={styles.account}>
              <Text numberOfLines={1} style={styles.title}>
                {title}
              </Text>
              {detail ? (
                <Text numberOfLines={1} style={styles.detail}>
                  {detail}
                </Text>
              ) : null}
            </View>
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
  button: {
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radius.pill,
    backgroundColor: colors.menuButton,
  },
  buttonShadow: shadows.menuButton,
  pressed: { opacity: 0.7 },
  account: { paddingHorizontal: spacing.sm, paddingTop: spacing.xs, paddingBottom: spacing.sm },
  title: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
  divider: {
    height: StyleSheet.hairlineWidth,
    marginVertical: spacing.xs,
    backgroundColor: colors.border,
  },
  menu: {
    position: "absolute",
    width: MENU_WIDTH,
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
