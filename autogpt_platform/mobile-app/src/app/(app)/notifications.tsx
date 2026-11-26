import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Pressable, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { Toast } from "@/components/Toast";
import { BriefingPicker } from "@/features/settings/components/BriefingPicker";
import { SettingsGroup } from "@/features/settings/components/SettingsGroup";
import { SettingsRow } from "@/features/settings/components/SettingsRow";
import { SwitchRow } from "@/features/settings/components/SwitchRow";
import { timezoneLabel } from "@/features/settings/notificationHelpers";
import { useNotificationsScreen } from "@/features/settings/useNotificationsScreen";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

/** What email the account gets, and the time zone briefings and routines keep. */
export default function NotificationsScreen() {
  const screen = useNotificationsScreen();
  const { preferences, zone, zoneOffer } = screen;
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();

  function renderBody() {
    if (preferences && zone !== null) {
      return (
        <ScrollView
          contentContainerStyle={[
            styles.content,
            { paddingBottom: insets.bottom + spacing.xxl },
          ]}
        >
          <SettingsGroup
            title="Email"
            note="Billing and account messages are always sent — they are about your account, not a promotion."
          >
            <BriefingPicker
              value={preferences.briefing_frequency}
              onChoose={screen.chooseBriefing}
            />
            <SwitchRow
              label="Alerts"
              description="Only when something is waiting on you — never for a successful run. At most two a day."
              value={preferences.alerts_enabled}
              onValueChange={screen.setAlerts}
            />
          </SettingsGroup>
          <SettingsGroup
            title="Time zone"
            note="Briefings and routines use this time zone."
          >
            <SettingsRow label="Time zone" value={timezoneLabel(zone)} />
            {zoneOffer ? (
              <Pressable
                accessibilityRole="button"
                onPress={() => screen.chooseZone(zoneOffer)}
                style={({ pressed }) => [styles.action, pressed && styles.pressed]}
              >
                <Text style={styles.actionText}>
                  {`Use this phone's time zone (${zoneOffer})`}
                </Text>
              </Pressable>
            ) : null}
          </SettingsGroup>
        </ScrollView>
      );
    }
    if (screen.isLoading) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>{"Your settings didn't load. Try again."}</Text>
        <Button label="Try again" variant="secondary" size="small" onPress={screen.retry} />
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        title="Notifications"
        left={<IconButton icon={ArrowLeft01Icon} label="Back" onPress={screen.goBack} />}
      />
      {renderBody()}
      {screen.toast ? <Toast toast={screen.toast} /> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.menuBackground },
  content: { gap: spacing.xl, padding: spacing.lg },
  loading: { marginTop: spacing.xxl },
  center: { alignItems: "center", gap: spacing.lg, padding: spacing.xl },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  action: {
    justifyContent: "center",
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  actionText: { ...type.body, color: colors.accentText },
}));
