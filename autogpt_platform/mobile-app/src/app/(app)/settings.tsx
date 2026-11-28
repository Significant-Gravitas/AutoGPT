import {
  Delete02Icon,
  File01Icon,
  HelpCircleIcon,
  Logout01Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { ScreenHeader } from "@/components/ScreenHeader";
import { AppearancePicker } from "@/features/settings/components/AppearancePicker";
import { DeleteAccountSheet } from "@/features/settings/components/DeleteAccountSheet/DeleteAccountSheet";
import { SettingsGroup } from "@/features/settings/components/SettingsGroup";
import { SettingsRow } from "@/features/settings/components/SettingsRow";
import { useSettingsScreen } from "@/features/settings/useSettingsScreen";
import { avatarSource } from "@/features/team/helpers";
import { createStyles, spacing, type } from "@/lib/theme";

export default function SettingsScreen() {
  const settings = useSettingsScreen();
  const styles = useStyles();
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.screen}>
      <ScreenHeader title="Settings" />
      <ScrollView
        contentContainerStyle={[
          styles.content,
          { paddingBottom: insets.bottom + spacing.xxl },
        ]}
      >
        <View style={styles.person}>
          <ExpertAvatar
            name={settings.name || settings.email}
            source={avatarSource({ avatar_url: settings.image })}
            size={56}
          />
          <View style={styles.personText}>
            {settings.name ? (
              <Text numberOfLines={1} style={styles.name}>
                {settings.name}
              </Text>
            ) : null}
            <Text numberOfLines={1} style={styles.email}>
              {settings.email}
            </Text>
          </View>
        </View>
        {settings.isPaymentEnabled ? (
          <SettingsGroup>
            <SettingsRow
              icon={StarIcon}
              label="Plan"
              value={settings.planLabel}
              trailing="page"
              onPress={settings.openPlan}
            />
          </SettingsGroup>
        ) : null}
        <SettingsGroup>
          <AppearancePicker />
        </SettingsGroup>
        <SettingsGroup>
          <SettingsRow icon={HelpCircleIcon} label="Help" trailing="web" onPress={settings.openHelp} />
          <SettingsRow icon={File01Icon} label="Terms of Use" trailing="web" onPress={settings.openTerms} />
          <SettingsRow icon={File01Icon} label="Privacy Policy" trailing="web" onPress={settings.openPrivacy} />
        </SettingsGroup>
        <SettingsGroup>
          <SettingsRow icon={Logout01Icon} label="Sign out" onPress={settings.signOut} />
          <SettingsRow
            icon={Delete02Icon}
            label="Delete account"
            isDestructive
            onPress={settings.openDelete}
          />
        </SettingsGroup>
        {settings.version ? (
          <Text style={styles.version}>AutoGPT {settings.version}</Text>
        ) : null}
      </ScrollView>
      <DeleteAccountSheet
        email={settings.email}
        isOpen={settings.isDeleteOpen}
        onClose={settings.closeDelete}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.menuBackground },
  content: { gap: spacing.xl, padding: spacing.lg },
  person: { flexDirection: "row", alignItems: "center", gap: spacing.md, paddingHorizontal: spacing.xs },
  personText: { flex: 1 },
  name: { ...type.heading, color: colors.text },
  email: { ...type.small, color: colors.textSecondary },
  version: { ...type.small, color: colors.textTertiary, textAlign: "center" },
}));
