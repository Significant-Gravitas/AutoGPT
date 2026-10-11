import { router, Stack } from "expo-router";
import { useState } from "react";
import { Alert, ScrollView, StyleSheet, Text } from "react-native";

import { Avatar } from "../../design/components/Avatar";
import { Row, Section } from "../../design/components/Grouped";
import { colors, spacing, type } from "../../design/theme";
import { useSession, useSignedIn } from "../../features/auth/SessionProvider";
import { hostLabel } from "../../features/servers/serverConfig";

export default function AccountScreen() {
  const { account, server } = useSignedIn();
  const { signOut } = useSession();
  const [signingOut, setSigningOut] = useState(false);

  function confirmSignOut() {
    Alert.alert(
      "Sign out?",
      `This removes ${account.email}'s sign-in from this device.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Sign Out",
          style: "destructive",
          onPress: () => {
            setSigningOut(true);
            void signOut();
          },
        },
      ],
    );
  }

  return (
    <>
      <Stack.Toolbar placement="right">
        <Stack.Toolbar.Button variant="done" onPress={() => router.back()}>
          Done
        </Stack.Toolbar.Button>
      </Stack.Toolbar>
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={styles.screen}
        contentContainerStyle={styles.content}
      >
        <Section>
          <Row
            title={account.name ?? account.email}
            subtitle={account.name ? account.email : null}
            leading={
              <Avatar
                name={account.name ?? account.email}
                uri={null}
                size={48}
              />
            }
          />
        </Section>
        <Section
          header="Server"
          footer="Your sign-in only goes to this server's sign-in address, and is kept in this device's keychain."
        >
          <Row
            title={server.name}
            subtitle={
              server.kind === "cloud" ? "Hosted by AutoGPT" : "Your own server"
            }
          />
          <Row title="Sign-in" subtitle={hostLabel(server.authUrl)} />
          <Row title="API" subtitle={hostLabel(server.apiUrl)} />
        </Section>
        <Section footer="To use a different account or server, sign out first.">
          <Row
            title={signingOut ? "Signing Out…" : "Sign Out"}
            destructive
            onPress={signingOut ? undefined : confirmSignOut}
            testID="sign-out"
          />
        </Section>
        <Text style={[type.footnote, styles.note]}>
          Connections, subscriptions and notifications aren&apos;t in the app
          yet.
        </Text>
      </ScrollView>
    </>
  );
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.groupedBackground },
  content: { paddingBottom: spacing.xxl },
  note: {
    color: colors.secondaryLabel,
    marginHorizontal: spacing.xl,
    marginTop: spacing.xl,
  },
});
