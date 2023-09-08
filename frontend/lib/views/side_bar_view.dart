import 'package:flutter/material.dart';

class SideBarView extends StatelessWidget {
  final ValueNotifier<String> selectedViewNotifier;

  SideBarView({required this.selectedViewNotifier});

  @override
  Widget build(BuildContext context) {
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
                    icon: Icon(Icons.chat),
                    onPressed: () => selectedViewNotifier.value = 'TaskView',
                  ),
                  IconButton(
                    splashRadius: 0.1,
                    color: selectedView == 'SkillTreeView'
                        ? Colors.blue
                        : Colors.black,
                    icon: Icon(Icons.emoji_events), // trophy icon
                    onPressed: () =>
                        selectedViewNotifier.value = 'SkillTreeView',
                  ),
                ],
              ),
            );
          }),
    );
  }
}
