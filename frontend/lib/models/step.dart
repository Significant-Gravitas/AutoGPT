// TODO: Refactor this to match which values are required and optional
class Step {
  final String input;
  final Map<String, dynamic> additionalInput;
  final String taskId;
  final String stepId;
  final String name;
  final String status;
  final String output;
  final Map<String, dynamic> additionalOutput;
  // TODO: Create an actual artifact object
  final List<dynamic> artifacts;
  final bool isLast;

  Step({
    required this.input,
    required this.additionalInput,
    required this.taskId,
    required this.stepId,
    required this.name,
    required this.status,
    required this.output,
    required this.additionalOutput,
    required this.artifacts,
    required this.isLast,
  });

  factory Step.fromMap(Map<String, dynamic>? map) {
    if (map == null) {
      throw ArgumentError('Null map provided to Step.fromMap');
    }
    return Step(
      input: map['input'] ?? '',
      additionalInput: map['additional_input'] != null
          ? Map<String, dynamic>.from(map['additional_input'])
          : {},
      taskId: map['task_id'] ?? '',
      stepId: map['step_id'] ?? '',
      name: map['name'] ?? '',
      status: map['status'] ?? '',
      output: map['output'] ?? '',
      additionalOutput: map['additional_output'] != null
          ? Map<String, dynamic>.from(map['additional_output'])
          : {},
      artifacts:
          map['artifacts'] != null ? List<dynamic>.from(map['artifacts']) : [],
      isLast: map['is_last'] ?? false,
    );
  }
}
