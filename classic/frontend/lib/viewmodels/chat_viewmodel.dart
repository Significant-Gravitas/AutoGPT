import 'package:auto_gpt_flutter_client/models/step.dart';
import 'package:auto_gpt_flutter_client/models/step_request_body.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:flutter/foundation.dart';
import 'package:auto_gpt_flutter_client/services/chat_service.dart';
import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';

class ChatViewModel with ChangeNotifier {
  final ChatService _chatService;
  List<Chat> _chats = [];
  String? _currentTaskId;
  final SharedPreferencesService _prefsService;

  bool _isWaitingForAgentResponse = false;

  bool get isWaitingForAgentResponse => _isWaitingForAgentResponse;
  SharedPreferencesService get prefsService => _prefsService;

  bool _isContinuousMode = false;

  bool get isContinuousMode => _isContinuousMode;
  set isContinuousMode(bool value) {
    _isContinuousMode = value;
    notifyListeners();
  }

  ChatViewModel(this._chatService, this._prefsService);

  /// Returns the current list of chats.
  List<Chat> get chats => _chats;

  String? get currentTaskId => _currentTaskId;

  void setCurrentTaskId(String taskId) {
    if (_currentTaskId != taskId) {
      _currentTaskId = taskId;
      fetchChatsForTask();
    }
  }

  void clearCurrentTaskAndChats() {
    _currentTaskId = null;
    _chats.clear();
    notifyListeners(); // Notify listeners to rebuild UI
  }

  /// Fetches chats from the data source for a specific task.
  void fetchChatsForTask() async {
    if (_currentTaskId == null) {
      print("Error: Task ID is not set.");
      return;
    }
    try {
      // Fetch task steps from the data source
      final Map<String, dynamic> stepsResponse =
          await _chatService.listTaskSteps(_currentTaskId!, pageSize: 10000);

      // Extract steps from the response
      final List<dynamic> stepsJsonList = stepsResponse['steps'] ?? [];

      // Convert each map into a Step object
      List<Step> steps =
          stepsJsonList.map((stepMap) => Step.fromMap(stepMap)).toList();

      // Initialize an empty list to store Chat objects
      List<Chat> chats = [];

      // Generate current timestamp
      DateTime currentTimestamp = DateTime.now();

      for (int i = 0; i < steps.length; i++) {
        Step step = steps[i];

        // Create a Chat object for 'input' if it exists and is not empty
        if (step.input.isNotEmpty) {
          chats.add(Chat(
              id: step.stepId,
              taskId: step.taskId,
              message: step.input,
              timestamp: currentTimestamp,
              messageType: MessageType.user,
              artifacts: step.artifacts));
        }

        // Create a Chat object for 'output'
        chats.add(Chat(
            id: step.stepId,
            taskId: step.taskId,
            message: step.output,
            timestamp: currentTimestamp,
            messageType: MessageType.agent,
            jsonResponse: stepsJsonList[i],
            artifacts: step.artifacts));
      }

      // Assign the chats list
      if (chats.length > 0) {
        _chats = chats;
      }

      // Notify listeners to rebuild UI
      notifyListeners();

      print(
          "Chats (and steps) fetched successfully for task ID: $_currentTaskId");
    } catch (error) {
      print("Error fetching chats: $error");
      // TODO: Handle additional error scenarios or log them as required
    }
  }

  /// Sends a chat message for a specific task.
  void sendChatMessage(String message,
      {required int continuousModeSteps, int currentStep = 1}) async {
    if (_currentTaskId == null) {
      print("Error: Task ID is not set.");
      return;
    }
    _isWaitingForAgentResponse = true;
    notifyListeners();

    try {
      // Create the request body for executing the step
      StepRequestBody requestBody = StepRequestBody(input: message);

      // Execute the step and get the response
      Map<String, dynamic> executedStepResponse =
          await _chatService.executeStep(_currentTaskId!, requestBody);

      // Create a Chat object from the returned step
      Step executedStep = Step.fromMap(executedStepResponse);

      // Create a Chat object for the user message
      if (executedStep.input.isNotEmpty) {
        final userChat = Chat(
            id: executedStep.stepId,
            taskId: executedStep.taskId,
            message: executedStep.input,
            timestamp: DateTime.now(),
            messageType: MessageType.user,
            artifacts: executedStep.artifacts);

        _chats.add(userChat);
      }

      // Create a Chat object for the agent message
      final agentChat = Chat(
          id: executedStep.stepId,
          taskId: executedStep.taskId,
          message: executedStep.output,
          timestamp: DateTime.now(),
          messageType: MessageType.agent,
          jsonResponse: executedStepResponse,
          artifacts: executedStep.artifacts);

      _chats.add(agentChat);

      // Remove the temporary message
      removeTemporaryMessage();

      // Notify UI of the new chats
      notifyListeners();

      if (_isContinuousMode && !executedStep.isLast) {
        print("Continuous Mode: Step $currentStep of $continuousModeSteps");
        if (currentStep < continuousModeSteps) {
          sendChatMessage("",
              continuousModeSteps: continuousModeSteps,
              currentStep: currentStep + 1);
        } else {
          _isContinuousMode = false;
        }
      }

      print("Chats added for task ID: $_currentTaskId");
    } catch (e) {
      // Remove the temporary message in case of an error
      removeTemporaryMessage();
      // TODO: We are bubbling up the full response. Revisit this.
      rethrow;
      // TODO: Handle additional error scenarios or log them as required
    } finally {
      _isWaitingForAgentResponse = false;
      notifyListeners();
    }
  }

  void addTemporaryMessage(String message) {
    Chat tempMessage = Chat(
        // You can generate a unique ID or use a placeholder
        id: "TEMP_ID",
        taskId: "TEMP_ID",
        message: message,
        timestamp: DateTime.now(),
        messageType: MessageType.user,
        artifacts: []);

    _chats.add(tempMessage);
    notifyListeners();
  }

  void removeTemporaryMessage() {
    _chats.removeWhere((chat) => chat.id == "TEMP_ID");
    notifyListeners();
  }

  /// Downloads an artifact associated with a specific chat.
  ///
  /// [taskId] is the ID of the task.
  /// [artifactId] is the ID of the artifact to be downloaded.
  Future<void> downloadArtifact(String taskId, String artifactId) async {
    try {
      // Call the downloadArtifact method from the ChatService class
      await _chatService.downloadArtifact(taskId, artifactId);

      print("Artifact $artifactId downloaded successfully for task $taskId!");
    } catch (error) {
      print("Error downloading artifact: $error");
      // TODO: Handle the error appropriately, perhaps notify the user
    }
  }
}
