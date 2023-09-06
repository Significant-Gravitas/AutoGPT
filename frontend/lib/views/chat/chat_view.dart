import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/chat/agent_message_tile.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_input_field.dart';
import 'package:auto_gpt_flutter_client/views/chat/user_message_tile.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:provider/provider.dart';

// TODO: Implement artifacts

class ChatView extends StatefulWidget {
  final ChatViewModel viewModel;

  const ChatView({Key? key, required this.viewModel}) : super(key: key);

  @override
  _ChatViewState createState() => _ChatViewState();
}

class _ChatViewState extends State<ChatView> {
  final ScrollController _scrollController = ScrollController();
  bool _isAtBottom = true;

  @override
  void initState() {
    super.initState();

    // Listen for scroll events and determine whether the scroll is at the bottom
    _scrollController.addListener(() {
      if (_scrollController.position.atEdge) {
        if (_scrollController.position.pixels == 0) {
          _isAtBottom = false;
        } else {
          _isAtBottom = true;
        }
      }
    });

    // Schedule the fetchTasks call for after the initial build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      widget.viewModel.fetchChatsForTask();
    });
  }

  @override
  void dispose() {
    // Dispose of the ScrollController when the widget is removed
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    _scrollController.animateTo(
      _scrollController.position.maxScrollExtent,
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    // TODO: Do we want to have a reference to task view model in this class?
    final taskViewModel = Provider.of<TaskViewModel>(context, listen: false);
    return Scaffold(
      body: Column(
        children: [
          // Chat messages list
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: widget.viewModel.chats.length,
              itemBuilder: (context, index) {
                final chat = widget.viewModel.chats[index];

                // If the last message has been built and we're at the bottom of the list, scroll down
                if (index == widget.viewModel.chats.length - 1 && _isAtBottom) {
                  WidgetsBinding.instance.addPostFrameCallback((_) {
                    _scrollToBottom();
                  });
                }

                if (chat.messageType == MessageType.user) {
                  return UserMessageTile(message: chat.message);
                } else {
                  return AgentMessageTile(chat: chat);
                }
              },
            ),
          ),
          // Input area
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ChatInputField(
              onSendPressed: (message) async {
                if (widget.viewModel.currentTaskId != null) {
                  widget.viewModel
                      .sendChatMessage((message == "") ? null : message);
                } else {
                  String newTaskId = await taskViewModel.createTask(message);
                  widget.viewModel.setCurrentTaskId(newTaskId);
                  widget.viewModel
                      .sendChatMessage((message == "") ? null : message);
                }
              },
              onContinuousModePressed: () {
                widget.viewModel.isContinuousMode =
                    !widget.viewModel.isContinuousMode;
              },
              isContinuousMode: widget.viewModel.isContinuousMode,
            ),
          ),
        ],
      ),
    );
  }
}
