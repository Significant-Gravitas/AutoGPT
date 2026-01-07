# Agent Executor

### What it is
Executes an existing agent inside your agent.

### What it does
Executes an existing agent inside your agent

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| user_id | User ID | str | Yes |
| graph_id | Graph ID | str | Yes |
| graph_version | Graph Version | int | Yes |
| agent_name | Name to display in the Builder UI | str | No |
| inputs | Input data for the graph | Dict[str, True] | Yes |
| input_schema | Input schema for the graph | Dict[str, True] | Yes |
| output_schema | Output schema for the graph | Dict[str, True] | Yes |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Execute Code

### What it is
Executes code in a sandbox environment with internet access.

### What it does
Executes code in a sandbox environment with internet access.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| setup_commands | Shell commands to set up the sandbox before running the code. You can use `curl` or `git` to install your desired Debian based package manager. `pip` and `npm` are pre-installed.

These commands are executed with `sh`, in the foreground. | List[str] | No |
| code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" | "js" | "bash" | No |
| timeout | Execution timeout in seconds | int | No |
| dispose_sandbox | Whether to dispose of the sandbox immediately after execution. If disabled, the sandbox will run until its timeout expires. | bool | No |
| template_id | You can use an E2B sandbox template by entering its ID here. Check out the E2B docs for more details: [E2B - Sandbox template](https://e2b.dev/docs/sandbox-template) | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| main_result | The main result from the code execution | Main Result |
| results | List of results from the code execution | List[CodeExecutionResult] |
| response | Text output (if any) of the main execution result | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Execute Code Step

### What it is
Execute code in a previously instantiated sandbox.

### What it does
Execute code in a previously instantiated sandbox.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the sandbox instance to execute the code in | str | Yes |
| step_code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" | "js" | "bash" | No |
| dispose_sandbox | Whether to dispose of the sandbox after executing this code. | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| main_result | The main result from the code execution | Main Result |
| results | List of results from the code execution | List[CodeExecutionResult] |
| response | Text output (if any) of the main execution result | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Reddit Posts

### What it is
This block fetches Reddit posts from a defined subreddit name.

### What it does
This block fetches Reddit posts from a defined subreddit name.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to Reddit using provided credentials, accesses the specified subreddit, and retrieves posts based on the given parameters. It can limit the number of posts, stop at a specific post, or fetch posts within a certain time frame.
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subreddit | Subreddit name, excluding the /r/ prefix | str | No |
| last_minutes | Post time to stop minutes ago while fetching posts | int | No |
| last_post | Post ID to stop when reached while fetching posts | str | No |
| post_limit | Number of posts to fetch | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post | Reddit post | RedditPost |
| posts | List of all Reddit posts | List[RedditPost] |

### Possible use case
<!-- MANUAL: use_case -->
A content curator could use this block to gather recent posts from a specific subreddit for analysis, summarization, or inclusion in a newsletter.
<!-- END MANUAL -->

---

## Instantiate Code Sandbox

### What it is
Instantiate a sandbox environment with internet access in which you can execute code with the Execute Code Step block.

### What it does
Instantiate a sandbox environment with internet access in which you can execute code with the Execute Code Step block.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| setup_commands | Shell commands to set up the sandbox before running the code. You can use `curl` or `git` to install your desired Debian based package manager. `pip` and `npm` are pre-installed.

These commands are executed with `sh`, in the foreground. | List[str] | No |
| setup_code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" | "js" | "bash" | No |
| timeout | Execution timeout in seconds | int | No |
| template_id | You can use an E2B sandbox template by entering its ID here. Check out the E2B docs for more details: [E2B - Sandbox template](https://e2b.dev/docs/sandbox-template) | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sandbox_id | ID of the sandbox instance | str |
| response | Text result (if any) of the setup code execution | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Post Reddit Comment

### What it is
This block posts a Reddit comment on a specified Reddit post.

### What it does
This block posts a Reddit comment on a specified Reddit post.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to Reddit using the provided credentials, locates the specified post, and then adds the given comment to that post.
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| data | Reddit comment | RedditComment | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comment_id | Posted comment ID | str |

### Possible use case
<!-- MANUAL: use_case -->
An automated moderation system could use this block to post pre-defined responses or warnings on Reddit posts that violate community guidelines.
<!-- END MANUAL -->

---

## Publish To Medium

### What it is
Publishes a post to Medium.

### What it does
Publishes a post to Medium.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| author_id | The Medium AuthorID of the user. You can get this by calling the /me endpoint of the Medium API.

curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" https://api.medium.com/v1/me" the response will contain the authorId field. | str | No |
| title | The title of your Medium post | str | Yes |
| content | The main content of your Medium post | str | Yes |
| content_format | The format of the content: 'html' or 'markdown' | str | Yes |
| tags | List of tags for your Medium post (up to 5) | List[str] | Yes |
| canonical_url | The original home of this content, if it was originally published elsewhere | str | No |
| publish_status | The publish status | "public" | "draft" | "unlisted" | Yes |
| license | The license of the post: 'all-rights-reserved', 'cc-40-by', 'cc-40-by-sa', 'cc-40-by-nd', 'cc-40-by-nc', 'cc-40-by-nc-nd', 'cc-40-by-nc-sa', 'cc-40-zero', 'public-domain' | str | No |
| notify_followers | Whether to notify followers that the user has published | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the post creation failed | str |
| post_id | The ID of the created Medium post | str |
| post_url | The URL of the created Medium post | str |
| published_at | The timestamp when the post was published | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Read RSS Feed

### What it is
Reads RSS feed entries from a given URL.

### What it does
Reads RSS feed entries from a given URL.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| rss_url | The URL of the RSS feed to read | str | Yes |
| time_period | The time period to check in minutes relative to the run block runtime, e.g. 60 would check for new entries in the last hour. | int | No |
| polling_rate | The number of seconds to wait between polling attempts. | int | Yes |
| run_continuously | Whether to run the block continuously or just once. | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| entry | The RSS item | RSSEntry |
| entries | List of all RSS entries | List[RSSEntry] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Authenticated Web Request

### What it is
Make an authenticated HTTP request with host-scoped credentials (JSON / form / multipart).

### What it does
Make an authenticated HTTP request with host-scoped credentials (JSON / form / multipart).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to send the request to | str | Yes |
| method | The HTTP method to use for the request | "GET" | "POST" | "PUT" | No |
| headers | The headers to include in the request | Dict[str, str] | No |
| json_format | If true, send the body as JSON (unless files are also present). | bool | No |
| body | Form/JSON body payload. If files are supplied, this must be a mapping of form‑fields. | Dict[str, True] | No |
| files_name | The name of the file field in the form data. | str | No |
| files | Mapping of *form field name* → Image url / path / base64 url. | List[str (file)] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Errors for all other exceptions | str |
| response | The response from the server | Response |
| client_error | Errors on 4xx status codes | Client Error |
| server_error | Errors on 5xx status codes | Server Error |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Email

### What it is
This block sends an email using the provided SMTP credentials.

### What it does
This block sends an email using the provided SMTP credentials.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| to_email | Recipient email address | str | Yes |
| subject | Subject of the email | str | Yes |
| body | Body of the email | str | Yes |
| config | SMTP Config | SMTP Config | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the email sending failed | str |
| status | Status of the email sending operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Web Request

### What it is
Make an HTTP request (JSON / form / multipart).

### What it does
Make an HTTP request (JSON / form / multipart).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to send the request to | str | Yes |
| method | The HTTP method to use for the request | "GET" | "POST" | "PUT" | No |
| headers | The headers to include in the request | Dict[str, str] | No |
| json_format | If true, send the body as JSON (unless files are also present). | bool | No |
| body | Form/JSON body payload. If files are supplied, this must be a mapping of form‑fields. | Dict[str, True] | No |
| files_name | The name of the file field in the form data. | str | No |
| files | Mapping of *form field name* → Image url / path / base64 url. | List[str (file)] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Errors for all other exceptions | str |
| response | The response from the server | Response |
| client_error | Errors on 4xx status codes | Client Error |
| server_error | Errors on 5xx status codes | Server Error |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Transcribe Youtube Video

### What it is
Transcribes a YouTube video using a proxy.

### What it does
Transcribes a YouTube video using a proxy.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| youtube_url | The URL of the YouTube video to transcribe | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Any error message if the transcription fails | str |
| video_id | The extracted YouTube video ID | str |
| transcript | The transcribed text of the video | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
