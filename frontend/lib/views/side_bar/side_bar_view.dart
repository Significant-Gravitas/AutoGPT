import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class SideBarView extends StatelessWidget {
  final ValueNotifier<String> selectedViewNotifier;

  const SideBarView({super.key, required this.selectedViewNotifier});

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
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  IconButton(
                    splashRadius: 0.1,
                    color:
                        selectedView == 'TaskView' ? Colors.blue : Colors.black,
                    icon: const Icon(Icons.chat),
                    onPressed: skillTreeViewModel.isBenchmarkRunning
                        ? null
                        : () => selectedViewNotifier.value = 'TaskView',
                  ),
                  IconButton(
                    splashRadius: 0.1,
                    color: selectedView == 'SkillTreeView'
                        ? Colors.blue
                        : Colors.black,
                    icon: const Icon(Icons.emoji_events), // trophy icon
                    onPressed: skillTreeViewModel.isBenchmarkRunning
                        ? null
                        : () => selectedViewNotifier.value = 'SkillTreeView',
                  ),
                ],
              ),
            );
          }),
    );
  }
}
