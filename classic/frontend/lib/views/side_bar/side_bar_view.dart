import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
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
    final taskQueueViewModel =
        Provider.of<TaskQueueViewModel>(context, listen: true);
    return Material(
      child: ValueListenableBuilder(
          valueListenable: selectedViewNotifier,
          builder: (context, String selectedView, _) {
            return SizedBox(
              width: 60,
              child: Column(
                children: [
                  Column(
                    children: [
                      IconButton(
                        splashRadius: 0.1,
                        color: selectedView == 'TaskView'
                            ? Colors.blue
                            : Colors.black,
                        icon: const Icon(Icons.chat),
                        onPressed: taskQueueViewModel.isBenchmarkRunning
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
                          onPressed: taskQueueViewModel.isBenchmarkRunning
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
                  const Spacer(),
                  Column(
                    children: [
                      IconButton(
                        splashRadius: 0.1,
                        iconSize: 25,
                        icon: Icon(Icons.book,
                            color: Color.fromRGBO(50, 120, 123, 1)),
                        onPressed: () => _launchURL(
                            'https://aiedge.medium.com/autogpt-forge-e3de53cc58ec'),
                        tooltip: 'Learn how to build your own Agent',
                      ),
                      IconButton(
                        splashRadius: 0.1,
                        iconSize: 25,
                        icon: Image.asset('assets/images/discord_logo.png'),
                        onPressed: () =>
                            _launchURL('https://discord.gg/autogpt'),
                        tooltip: 'Join our Discord',
                      ),
                      const SizedBox(height: 6),
                      IconButton(
                        splashRadius: 0.1,
                        iconSize: 15,
                        icon: Image.asset('assets/images/twitter_logo.png'),
                        onPressed: () =>
                            _launchURL('https://twitter.com/Auto_GPT'),
                        tooltip: 'Follow us on Twitter',
                      ),
                      const SizedBox(height: 8),
                    ],
                  ),
                ],
              ),
            );
          }),
    );
  }
}
