import { InformationCircleIcon, LinkSquare02Icon, SquareLock02Icon } from "@hugeicons/core-free-icons";
import { StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { useNoPlanScreen } from "@/features/plan/useNoPlanScreen";
import { avatarSource } from "@/features/team/helpers";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

/** An account with no plan stops here: no way past, and no way to buy in the app. */
export default function NoPlanScreen() {
  const {
    email,
    name,
    image,
    isChecking,
    notice,
    openSignup,
    checkAgain,
    signOut,
  } = useNoPlanScreen();
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();

  return (
    <View
      style={[
        styles.screen,
        { paddingTop: insets.top, paddingBottom: insets.bottom + spacing.lg },
      ]}
    >
      <View style={styles.middle}>
        <View style={styles.mark}>
          <Icon icon={SquareLock02Icon} size={28} color={colors.textSecondary} />
        </View>
        <Text accessibilityRole="header" style={styles.heading}>
          This account has no active plan
        </Text>
        <Text style={styles.body}>
          Get a subscription, then come back to the app.
        </Text>
        <View style={styles.account}>
          <ExpertAvatar
            name={name || email}
            source={avatarSource({ avatar_url: image })}
            size={28}
          />
          <Text numberOfLines={1} style={styles.email}>
            {email}
          </Text>
        </View>
      </View>
      <View style={styles.footer}>
        {notice ? (
          <View accessibilityLiveRegion="polite" style={styles.notice}>
            <Icon icon={InformationCircleIcon} size={16} color={colors.textSecondary} />
            <Text style={styles.noticeText}>{notice}</Text>
          </View>
        ) : null}
        <Button
          label="Get a subscription"
          leadingIcon={LinkSquare02Icon}
          onPress={openSignup}
          fullWidth
        />
        <Button
          label={isChecking ? "Checking…" : "Check again"}
          variant="secondary"
          onPress={checkAgain}
          isLoading={isChecking}
          fullWidth
        />
        <Button label="Sign out" variant="ghost" onPress={signOut} fullWidth />
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: {
    flex: 1,
    paddingHorizontal: spacing.xl,
    backgroundColor: colors.background,
  },
  middle: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.md,
  },
  mark: {
    width: 64,
    height: 64,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.xs,
    borderRadius: 20,
    backgroundColor: colors.surfaceRaised,
  },
  heading: { ...type.headline, color: colors.text, textAlign: "center" },
  body: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  account: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    maxWidth: "100%",
    marginTop: spacing.xs,
    paddingVertical: spacing.xs,
    paddingLeft: spacing.xs,
    paddingRight: spacing.md,
    borderRadius: radius.pill,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  email: { ...type.small, flexShrink: 1, color: colors.text },
  footer: { gap: spacing.sm },
  notice: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  noticeText: { ...type.small, flex: 1, color: colors.textSecondary },
}));
