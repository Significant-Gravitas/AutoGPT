import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

class SideBarView extends StatelessWidget {
  final ValueNotifier<String> selectedViewNotifier;

  const SideBarView({super.key, required this.selectedViewNotifier});

  // Function to launch the URL
  void _launchURL(String urlString) async {
    var url = Uri.parse(urlString);
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  @override
  Widget build(BuildContext context) {
    // TODO: should we pass this in as a dependency?
    final skillTreeViewModel =
        Provider.of<SkillTreeViewModel>(context, listen: true);
    return Material(
      child: ValueListenableBuilder(
          valueListenable: selectedViewNotifier,
          builder: (context, String selectedView, _) {
            return SizedBox(
              width: 60,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    children: [
                      IconButton(
                        splashRadius: 0.1,
                        color: selectedView == 'TaskView'
                            ? Colors.blue
                            : Colors.black,
                        icon: const Icon(Icons.chat),
                        onPressed: skillTreeViewModel.isBenchmarkRunning
                            ? null
                            : () => selectedViewNotifier.value = 'TaskView',
                      ),
                      if (Provider.of<SettingsViewModel>(context, listen: true)
                          .isDeveloperModeEnabled)
                        IconButton(
                          splashRadius: 0.1,
                          color: selectedView == 'SkillTreeView'
                              ? Colors.blue
                              : Colors.black,
                          icon: const Icon(Icons.emoji_events),
                          onPressed: skillTreeViewModel.isBenchmarkRunning
                              ? null
                              : () =>
                                  selectedViewNotifier.value = 'SkillTreeView',
                        ),
                      IconButton(
                        splashRadius: 0.1,
                        color: selectedView == 'SettingsView'
                            ? Colors.blue
                            : Colors.black,
                        icon: const Icon(Icons.settings),
                        onPressed: () =>
                            selectedViewNotifier.value = 'SettingsView',
                      ),
                    ],
                  ),
                  Column(
                    children: [
                      IconButton(
                        splashRadius: 0.1,
                        icon: Image.asset('assets/images/autogpt_logo.png'),
                        onPressed: () =>
                            _launchURL('https://leaderboard.agpt.co'),
                        tooltip: 'Check out the leaderboard',
                      ),
                      IconButton(
                        splashRadius: 0.1,
                        icon: Image.asset('assets/images/discord_logo.png'),
                        onPressed: () =>
                            _launchURL('https://discord.gg/autogpt'),
                        tooltip: 'Join our Discord',
                      ),
                      IconButton(
                        splashRadius: 0.1,
                        icon: Image.asset('assets/images/twitter_logo.png'),
                        onPressed: () =>
                            _launchURL('https://twitter.com/Auto_GPT'),
                        tooltip: 'Follow us on Twitter',
                      ),
                    ],
                  ),
                ],
              ),
            );
          }),
    );
  }
}
