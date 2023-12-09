/// Represents a task or topic the user wants to discuss with the agent.
class Task {
  final String id;
  final Map<String, dynamic>? additionalInput;
  final List<String>? artifacts;

  String _title;

  Task({
    required this.id,
    this.additionalInput,
    this.artifacts,
    required String title,
  })  : assert(title.isNotEmpty, 'Title cannot be empty'),
        _title = title;

  String get title => _title;

  set title(String newTitle) {
    if (newTitle.isNotEmpty) {
      _title = newTitle;
    } else {
      throw ArgumentError('Title cannot be empty.');
    }
  }

// Convert a Map (usually from JSON) to a Task object
  factory Task.fromMap(Map<String, dynamic> map) {
    Map<String, dynamic>? additionalInput;
    List<String>? artifacts;

    if (map['additional_input'] != null) {
      additionalInput = Map<String, dynamic>.from(map['additional_input']);
    }

    if (map['artifacts'] != null) {
      artifacts = List<String>.from(map['artifacts'].map((e) => e.toString()));
    }

    return Task(
      id: map['task_id'],
      additionalInput: additionalInput,
      artifacts: artifacts,
      title: map['input'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'task_id': id,
      'input': title,
      'additional_input': additionalInput,
      'artifacts': artifacts,
    };
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Task && runtimeType == other.runtimeType && id == other.id;

  @override
  int get hashCode => id.hashCode ^ title.hashCode;

  @override
  String toString() => 'Task(id: $id, title: $title)';
}
