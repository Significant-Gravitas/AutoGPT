import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/settings/settings_view.dart';
import 'package:auto_gpt_flutter_client/views/side_bar/side_bar_view.dart';
import 'package:auto_gpt_flutter_client/views/skill_tree/skill_tree_view.dart';
import 'package:auto_gpt_flutter_client/views/task/task_view.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_view.dart';
import 'package:auto_gpt_flutter_client/views/task_queue/task_queue_view.dart';
import 'package:flutter/cupertino.dart';
import 'package:provider/provider.dart';

class MainLayout extends StatelessWidget {
  final ValueNotifier<String> selectedViewNotifier = ValueNotifier('TaskView');

  MainLayout({super.key});

  @override
  Widget build(BuildContext context) {
    // Get the screen width
    double width = MediaQuery.of(context).size.width;

    // Access the various ViewModels from the context
    final taskViewModel = Provider.of<TaskViewModel>(context);
    final chatViewModel = Provider.of<ChatViewModel>(context);
    final settingsViewModel = Provider.of<SettingsViewModel>(context);

    // Initialize the width for the SideBarView
    double sideBarWidth = 60.0;

    // Initialize the width for the TaskView
    double taskViewWidth = 280.0;

    // Initialize the width for the SettingsView
    double settingsViewWidth = 280.0;

    // Calculate remaining width after subtracting the width of the SideBarView
    double remainingWidth = width - sideBarWidth;

    // Declare variables to hold the widths of SkillTreeView, TestQueueView, and ChatView
    double skillTreeViewWidth = 0;
    double testQueueViewWidth = 0;
    double chatViewWidth = 0;

    if (width > 800) {
      return Row(
        children: [
          SizedBox(
              width: sideBarWidth,
              child: SideBarView(selectedViewNotifier: selectedViewNotifier)),
          ValueListenableBuilder(
            valueListenable: selectedViewNotifier,
            builder: (context, String value, _) {
              return Consumer<SkillTreeViewModel>(
                builder: (context, skillTreeViewModel, _) {
                  if (value == 'TaskView') {
                    // TODO: Handle this state reset better
                    skillTreeViewModel.resetState();
                    chatViewWidth = remainingWidth - taskViewWidth;
                    return Row(
                      children: [
                        SizedBox(
                            width: taskViewWidth,
                            child: TaskView(viewModel: taskViewModel)),
                        SizedBox(
                            width: chatViewWidth,
                            child: ChatView(viewModel: chatViewModel))
                      ],
                    );
                  } else if (value == 'SettingsView') {
                    // TODO: Handle this state reset better
                    skillTreeViewModel.resetState();
                    chatViewWidth = remainingWidth - settingsViewWidth;
                    return Row(
                      children: [
                        SizedBox(
                            width: settingsViewWidth,
                            // Render the SettingsView with the same width as TaskView
                            child: SettingsView(viewModel: settingsViewModel)),
                        SizedBox(
                            width: chatViewWidth,
                            // Render the ChatView next to the SettingsView
                            child: ChatView(viewModel: chatViewModel)),
                      ],
                    );
                  } else {
                    if (skillTreeViewModel.selectedNode != null) {
                      // If TaskQueueView should be displayed
                      testQueueViewWidth = remainingWidth * 0.25;
                      skillTreeViewWidth = remainingWidth * 0.25;
                      chatViewWidth = remainingWidth * 0.5;
                    } else {
                      // If only SkillTreeView and ChatView should be displayed
                      skillTreeViewWidth = remainingWidth * 0.5;
                      chatViewWidth = remainingWidth * 0.5;
                    }

                    return Row(
                      children: [
                        SizedBox(
                            width: skillTreeViewWidth,
                            child:
                                SkillTreeView(viewModel: skillTreeViewModel)),
                        if (skillTreeViewModel.selectedNode != null)
                          SizedBox(
                              width: testQueueViewWidth,
                              child: TaskQueueView()),
                        SizedBox(
                            width: chatViewWidth,
                            child: ChatView(viewModel: chatViewModel)),
                      ],
                    );
                  }
                },
              );
            },
          ),
        ],
      );
    } else {
      // For smaller screens, return a tabbed layout
      // TODO: Include settings view for smaller screen sizes
      return CupertinoTabScaffold(
        tabBar: CupertinoTabBar(
          items: const <BottomNavigationBarItem>[
            BottomNavigationBarItem(
              icon: Icon(CupertinoIcons.person),
              label: 'Tasks',
            ),
            BottomNavigationBarItem(
              icon: Icon(CupertinoIcons.chat_bubble),
              label: 'Chat',
            ),
          ],
        ),
        tabBuilder: (BuildContext context, int index) {
          CupertinoTabView? returnValue;

          switch (index) {
            case 0:
              returnValue = CupertinoTabView(builder: (context) {
                return CupertinoPageScaffold(
                  child: SafeArea(child: TaskView(viewModel: taskViewModel)),
                );
              });
              break;
            case 1:
              returnValue = CupertinoTabView(builder: (context) {
                return CupertinoPageScaffold(
                  child: SafeArea(child: ChatView(viewModel: chatViewModel)),
                );
              });
              break;
          }

          return returnValue ??
              CupertinoTabView(builder: (context) {
                return CupertinoPageScaffold(
                  child: Container(), // Default empty container
                );
              });
        },
      );
    }
  }
}
