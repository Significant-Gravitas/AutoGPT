# Todoist Blocks

## Todoist Create Label

### What it is
A block that creates a new label in Todoist.

### What it does
Creates a new label in Todoist with specified name, order, color and favorite status.

### How it works
It takes label details as input, connects to Todoist API, creates the label and returns the created label's details.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Name | Name of the label |
| Order | Optional label order |
| Color | Optional color of the label icon |
| Is Favorite | Whether label is marked as favorite |

### Outputs
| Output | Description |
|--------|-------------|
| ID | ID of the created label |
| Name | Name of the label |
| Color | Color of the label |
| Order | Label order |
| Is Favorite | Favorite status |
| Error | Error message if request failed |

### Possible use case
Creating new labels to organize and categorize tasks in Todoist.

---

## Todoist List Labels

### What it is
A block that retrieves all personal labels from Todoist.

### What it does
Fetches all personal labels from the user's Todoist account.

### How it works
Connects to Todoist API using provided credentials and retrieves all labels.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |

### Outputs
| Output | Description |
|--------|-------------|
| Labels | List of complete label data |
| Label IDs | List of label IDs |
| Label Names | List of label names |
| Error | Error message if request failed |

### Possible use case
Getting an overview of all labels to organize tasks or find specific labels.

---

## Todoist Get Label

### What it is
A block that retrieves a specific label by ID.

### What it does
Fetches details of a specific label using its ID.

### How it works
Uses the label ID to retrieve label details from Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Label ID | ID of label to retrieve |

### Outputs
| Output | Description |
|--------|-------------|
| ID | Label ID |
| Name | Label name |
| Color | Label color |
| Order | Label order |
| Is Favorite | Favorite status |
| Error | Error message if request failed |

### Possible use case
Looking up details of a specific label for editing or verification.

---

## Todoist Create Task

### What it is
A block that creates a new task in Todoist.

### What it does
Creates a new task with specified content, description, project assignment and other optional parameters.

### How it works
Takes task details and creates a new task via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Content | Task content |
| Description | Optional task description |
| Project ID | Optional project to add task to |
| Section ID | Optional section to add task to |
| Parent ID | Optional parent task ID |
| Order | Optional task order |
| Labels | Optional task labels |
| Priority | Optional priority (1-4) |
| Due Date | Optional due date |
| Deadline Date | Optional deadline date |
| Assignee ID | Optional assignee |
| Duration Unit | Optional duration unit |
| Duration | Optional duration amount |

### Outputs
| Output | Description |
|--------|-------------|
| ID | Created task ID |
| URL | Task URL |
| Complete Data | Complete task data |
| Error | Error message if request failed |

### Possible use case
Creating new tasks with full customization of parameters.

---

## Todoist Get Tasks

### What it is
A block that retrieves active tasks from Todoist.

### What it does
Fetches tasks based on optional filters like project, section, label etc.

### How it works
Queries Todoist API with provided filters to get matching tasks.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Project ID | Optional filter by project |
| Section ID | Optional filter by section |
| Label | Optional filter by label |
| Filter | Optional custom filter string |
| Lang | Optional filter language |
| IDs | Optional specific task IDs |

### Outputs
| Output | Description |
|--------|-------------|
| IDs | List of task IDs |
| URLs | List of task URLs |
| Complete Data | Complete task data |
| Error | Error message if request failed |

### Possible use case
Retrieving tasks matching specific criteria for review or processing.

---

## Todoist Update Task

### What it is
A block that updates an existing task.

### What it does
Updates specified fields of an existing task.

### How it works
Takes task ID and updated fields, applies changes via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Task ID | ID of task to update |
| Content | New task content |
| Description | New description |
| Project ID | New project ID |
| Section ID | New section ID |
| Parent ID | New parent task ID |
| Order | New order |
| Labels | New labels |
| Priority | New priority |
| Due Date | New due date |
| Deadline Date | New deadline date |
| Assignee ID | New assignee |
| Duration Unit | New duration unit |
| Duration | New duration |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether update succeeded |
| Error | Error message if failed |

### Possible use case
Modifying task details like due dates, priority etc.

---

## Todoist Close Task

### What it is
A block that completes/closes a task.

### What it does
Marks a task as complete in Todoist.

### How it works
Uses task ID to mark it complete via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Task ID | ID of task to close |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether task was closed |
| Error | Error message if failed |

### Possible use case
Marking tasks as done in automated workflows.

---

## Todoist Reopen Task

### What it is
A block that reopens a completed task.

### What it does
Marks a completed task as active again.

### How it works
Uses task ID to reactivate via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Task ID | ID of task to reopen |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether task was reopened |
| Error | Error message if failed |

### Possible use case
Reactivating tasks that were closed accidentally or need to be repeated.

---

## Todoist Delete Task

### What it is
A block that permanently deletes a task.

### What it does
Removes a task completely from Todoist.

### How it works
Uses task ID to delete via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Task ID | ID of task to delete |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether deletion succeeded |
| Error | Error message if failed |

### Possible use case
Removing unwanted or obsolete tasks from the system.

---

## Todoist List Projects

### What it is
A block that retrieves all projects from Todoist.

### What it does
Fetches all projects and their details from a user's Todoist account.

### How it works
Connects to Todoist API using provided credentials and retrieves all projects.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |

### Outputs
| Output | Description |
|--------|-------------|
| Names List | List of project names |
| IDs List | List of project IDs |
| URL List | List of project URLs |
| Complete Data | Complete project data |
| Error | Error message if request failed |

### Possible use case
Getting an overview of all projects for organization or automation.

---

## Todoist Create Project

### What it is
A block that creates a new project in Todoist.

### What it does
Creates a new project with specified name, parent project, color and other settings.

### How it works
Takes project details and creates via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Name | Name of the project |
| Parent ID | Optional parent project ID |
| Color | Optional color of project icon |
| Is Favorite | Whether project is favorite |
| View Style | Display style (list/board) |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether creation succeeded |
| Error | Error message if failed |

### Possible use case
Creating new projects programmatically for workflow automation.

---

## Todoist Get Project

### What it is
A block that retrieves details for a specific project.

### What it does
Fetches complete details of a single project by ID.

### How it works
Uses project ID to retrieve details via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Project ID | ID of project to get |

### Outputs
| Output | Description |
|--------|-------------|
| Project ID | ID of the project |
| Project Name | Name of the project |
| Project URL | URL of the project |
| Complete Data | Complete project data |
| Error | Error message if failed |

### Possible use case
Looking up project details for verification or editing.

---

## Todoist Update Project

### What it is
A block that updates an existing project.

### What it does
Updates specified fields of an existing project.

### How it works
Takes project ID and updated fields, applies via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Project ID | ID of project to update |
| Name | New project name |
| Color | New color for icon |
| Is Favorite | New favorite status |
| View Style | New display style |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether update succeeded |
| Error | Error message if failed |

### Possible use case
Modifying project settings or reorganizing projects.

---

## Todoist Delete Project

### What it is
A block that deletes a project and its contents.

### What it does
Permanently removes a project including sections and tasks.

### How it works
Uses project ID to delete via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Project ID | ID of project to delete |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether deletion succeeded |
| Error | Error message if failed |

### Possible use case
Removing completed or obsolete projects.

---

## Todoist List Collaborators

### What it is
A block that retrieves collaborators on a project.

### What it does
Fetches all collaborators and their details for a specific project.

### How it works
Uses project ID to get collaborator list via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Project ID | ID of project to check |

### Outputs
| Output | Description |
|--------|-------------|
| Collaborator IDs | List of collaborator IDs |
| Collaborator Names | List of collaborator names |
| Collaborator Emails | List of collaborator emails |
| Complete Data | Complete collaborator data |
| Error | Error message if failed |

### Possible use case
Managing project sharing and collaboration.

---

## Todoist List Sections

### What it is
A block that retrieves sections from Todoist.

### What it does
Fetches all sections, optionally filtered by project.

### How it works
Connects to Todoist API to retrieve sections list.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Project ID | Optional project filter |

### Outputs
| Output | Description |
|--------|-------------|
| Names List | List of section names |
| IDs List | List of section IDs |
| Complete Data | Complete section data |
| Error | Error message if failed |

### Possible use case
Getting section information for task organization.

---

## Todoist Get Section

### What it is
A block that retrieves details for a specific section.

### What it does
Fetches complete details of a single section by ID.

### How it works
Uses section ID to retrieve details via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Section ID | ID of section to get |

### Outputs
| Output | Description |
|--------|-------------|
| ID | Section ID |
| Project ID | Parent project ID |
| Order | Section order |
| Name | Section name |
| Error | Error message if failed |

### Possible use case
Looking up section details for task management.

---

## Todoist Delete Section

### What it is
A block that deletes a section and its tasks.

### What it does
Permanently removes a section including all tasks.

### How it works
Uses section ID to delete via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Section ID | ID of section to delete |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether deletion succeeded |
| Error | Error message if failed |

### Possible use case
Removing unused sections or reorganizing projects.

---

## Todoist Create Comment

### What it is
A block that creates a new comment on a Todoist task or project.

### What it does
Creates a comment with specified content on either a task or project.

### How it works
Takes comment content and task/project ID, creates comment via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Content | Comment content |
| ID Type | Task ID or Project ID to comment on |
| Attachment | Optional file attachment |

### Outputs
| Output | Description |
|--------|-------------|
| ID | ID of created comment |
| Content | Comment content |
| Posted At | Comment timestamp |
| Task ID | Associated task ID |
| Project ID | Associated project ID |
| Error | Error message if request failed |

### Possible use case
Adding notes and comments to tasks or projects automatically.

---

## Todoist Get Comments

### What it is
A block that retrieves all comments for a task or project.

### What it does
Fetches all comments associated with a specific task or project.

### How it works
Uses task/project ID to get comments list via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| ID Type | Task ID or Project ID to get comments for |

### Outputs
| Output | Description |
|--------|-------------|
| Comments | List of comments |
| Error | Error message if request failed |

### Possible use case
Reviewing comment history on tasks or projects.

---

## Todoist Get Comment

### What it is
A block that retrieves a specific comment by ID.

### What it does
Fetches details of a single comment using its ID.

### How it works
Uses comment ID to retrieve details via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Comment ID | ID of comment to retrieve |

### Outputs
| Output | Description |
|--------|-------------|
| Content | Comment content |
| ID | Comment ID |
| Posted At | Comment timestamp |
| Project ID | Associated project ID |
| Task ID | Associated task ID |
| Attachment | Optional file attachment |
| Error | Error message if request failed |

### Possible use case
Looking up specific comment details for reference.

---

## Todoist Update Comment

### What it is
A block that updates an existing comment.

### What it does
Updates the content of a specific comment.

### How it works
Takes comment ID and new content, updates via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Comment ID | ID of comment to update |
| Content | New content for the comment |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether update succeeded |
| Error | Error message if request failed |

### Possible use case
Modifying existing comments to fix errors or update information.

---

## Todoist Delete Comment

### What it is
A block that deletes a comment.

### What it does
Permanently removes a comment from a task or project.

### How it works
Uses comment ID to delete via Todoist API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Todoist API credentials |
| Comment ID | ID of comment to delete |

### Outputs
| Output | Description |
|--------|-------------|
| Success | Whether deletion succeeded |
| Error | Error message if request failed |

### Possible use case
Removing outdated or incorrect comments from tasks/projects.
