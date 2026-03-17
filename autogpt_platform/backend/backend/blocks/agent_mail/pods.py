"""
AgentMail Pod blocks — create, get, list, delete pods and list pod-scoped resources.

Pods provide multi-tenant isolation between your customers. Each pod acts as
an isolated workspace containing its own inboxes, domains, threads, and drafts.
Use pods when building SaaS platforms, agency tools, or AI agent fleets that
serve multiple customers.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, _client, agent_mail


class AgentMailCreatePodBlock(Block):
    """
    Create a new pod for multi-tenant customer isolation.

    Each pod acts as an isolated workspace for one customer or tenant.
    Use client_id to map pods to your internal tenant IDs for idempotent
    creation (safe to retry without creating duplicates).
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        client_id: str = SchemaField(
            description="Your internal tenant/customer ID for idempotent mapping. Lets you access the pod by your own ID instead of AgentMail's pod_id.",
            default="",
        )

    class Output(BlockSchemaOutput):
        pod_id: str = SchemaField(description="Unique identifier of the created pod")
        result: dict = SchemaField(description="Complete pod object with all metadata")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="a2db9784-2d17-4f8f-9d6b-0214e6f22101",
            description="Create a new pod for multi-tenant customer isolation. Use client_id to map to your internal tenant IDs.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("pod_id", "mock-pod-id"),
                ("result", dict),
            ],
            test_mock={
                "create_pod": lambda *a, **kw: type(
                    "Pod",
                    (),
                    {
                        "pod_id": "mock-pod-id",
                        "model_dump": lambda self: {"pod_id": "mock-pod-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def create_pod(credentials: APIKeyCredentials, **params):
        client = _client(credentials)
        return await client.pods.create(**params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {}
            if input_data.client_id:
                params["client_id"] = input_data.client_id

            pod = await self.create_pod(credentials, **params)
            result = pod.model_dump()

            yield "pod_id", pod.pod_id
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailGetPodBlock(Block):
    """
    Retrieve details of an existing pod by its ID.

    Returns the pod metadata including its client_id mapping and
    creation timestamp.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        pod_id: str = SchemaField(description="Pod ID to retrieve")

    class Output(BlockSchemaOutput):
        pod_id: str = SchemaField(description="Unique identifier of the pod")
        result: dict = SchemaField(description="Complete pod object with all metadata")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="553361bc-bb1b-4322-9ad4-0c226200217e",
            description="Retrieve details of an existing pod including its client_id mapping and metadata.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "pod_id": "test-pod"},
            test_output=[
                ("pod_id", "test-pod"),
                ("result", dict),
            ],
            test_mock={
                "get_pod": lambda *a, **kw: type(
                    "Pod",
                    (),
                    {
                        "pod_id": "test-pod",
                        "model_dump": lambda self: {"pod_id": "test-pod"},
                    },
                )(),
            },
        )

    @staticmethod
    async def get_pod(credentials: APIKeyCredentials, pod_id: str):
        client = _client(credentials)
        return await client.pods.get(pod_id=pod_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            pod = await self.get_pod(credentials, pod_id=input_data.pod_id)
            result = pod.model_dump()

            yield "pod_id", pod.pod_id
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailListPodsBlock(Block):
    """
    List all pods in your AgentMail organization.

    Returns a paginated list of all tenant pods with their metadata.
    Use this to see all customer workspaces at a glance.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        limit: int = SchemaField(
            description="Maximum number of pods to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        pods: list[dict] = SchemaField(
            description="List of pod objects with pod_id, client_id, creation time, etc."
        )
        count: int = SchemaField(description="Number of pods returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="9d3725ee-2968-431a-a816-857ab41e1420",
            description="List all tenant pods in your organization. See all customer workspaces at a glance.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("pods", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_pods": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "pods": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_pods(credentials: APIKeyCredentials, **params):
        client = _client(credentials)
        return await client.pods.list(**params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token

            response = await self.list_pods(credentials, **params)
            pods = [p.model_dump() for p in response.pods]

            yield "pods", pods
            yield "count", response.count
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailDeletePodBlock(Block):
    """
    Permanently delete a pod. All inboxes and domains must be removed first.

    You cannot delete a pod that still contains inboxes or domains.
    Delete all child resources first, then delete the pod.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        pod_id: str = SchemaField(
            description="Pod ID to permanently delete (must have no inboxes or domains)"
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the pod was successfully deleted"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="f371f8cd-682d-4f5f-905c-529c74a8fb35",
            description="Permanently delete a pod. All inboxes and domains must be removed first.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "pod_id": "test-pod"},
            test_output=[("success", True)],
            test_mock={
                "delete_pod": lambda *a, **kw: None,
            },
        )

    @staticmethod
    async def delete_pod(credentials: APIKeyCredentials, pod_id: str):
        client = _client(credentials)
        await client.pods.delete(pod_id=pod_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            await self.delete_pod(credentials, pod_id=input_data.pod_id)
            yield "success", True
        except Exception as e:
            yield "error", str(e)


class AgentMailListPodInboxesBlock(Block):
    """
    List all inboxes within a specific pod (customer workspace).

    Returns only the inboxes belonging to this pod, providing
    tenant-scoped visibility.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        pod_id: str = SchemaField(description="Pod ID to list inboxes from")
        limit: int = SchemaField(
            description="Maximum number of inboxes to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        inboxes: list[dict] = SchemaField(
            description="List of inbox objects within this pod"
        )
        count: int = SchemaField(description="Number of inboxes returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="a8c17ce0-b7c1-4bc3-ae39-680e1952e5d0",
            description="List all inboxes within a pod. View email accounts scoped to a specific customer.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "pod_id": "test-pod"},
            test_output=[
                ("inboxes", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_pod_inboxes": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "inboxes": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_pod_inboxes(credentials: APIKeyCredentials, pod_id: str, **params):
        client = _client(credentials)
        return await client.pods.inboxes.list(pod_id=pod_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token

            response = await self.list_pod_inboxes(
                credentials, pod_id=input_data.pod_id, **params
            )
            inboxes = [i.model_dump() for i in response.inboxes]

            yield "inboxes", inboxes
            yield "count", response.count
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailListPodThreadsBlock(Block):
    """
    List all conversation threads across all inboxes within a pod.

    Returns threads from every inbox in the pod. Use for building
    per-customer dashboards showing all email activity, or for
    supervisor agents monitoring a customer's conversations.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        pod_id: str = SchemaField(description="Pod ID to list threads from")
        limit: int = SchemaField(
            description="Maximum number of threads to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Only return threads matching ALL of these labels",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        threads: list[dict] = SchemaField(
            description="List of thread objects from all inboxes in this pod"
        )
        count: int = SchemaField(description="Number of threads returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="80214f08-8b85-4533-a6b8-f8123bfcb410",
            description="List all conversation threads across all inboxes within a pod. View all email activity for a customer.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "pod_id": "test-pod"},
            test_output=[
                ("threads", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_pod_threads": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "threads": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_pod_threads(credentials: APIKeyCredentials, pod_id: str, **params):
        client = _client(credentials)
        return await client.pods.threads.list(pod_id=pod_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token
            if input_data.labels:
                params["labels"] = input_data.labels

            response = await self.list_pod_threads(
                credentials, pod_id=input_data.pod_id, **params
            )
            threads = [t.model_dump() for t in response.threads]

            yield "threads", threads
            yield "count", response.count
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailListPodDraftsBlock(Block):
    """
    List all drafts across all inboxes within a pod.

    Returns pending drafts from every inbox in the pod. Use for
    per-customer approval dashboards or monitoring scheduled sends.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        pod_id: str = SchemaField(description="Pod ID to list drafts from")
        limit: int = SchemaField(
            description="Maximum number of drafts to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        drafts: list[dict] = SchemaField(
            description="List of draft objects from all inboxes in this pod"
        )
        count: int = SchemaField(description="Number of drafts returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="12fd7a3e-51ad-4b20-97c1-0391f207f517",
            description="List all drafts across all inboxes within a pod. View pending emails for a customer.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "pod_id": "test-pod"},
            test_output=[
                ("drafts", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_pod_drafts": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "drafts": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_pod_drafts(credentials: APIKeyCredentials, pod_id: str, **params):
        client = _client(credentials)
        return await client.pods.drafts.list(pod_id=pod_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token

            response = await self.list_pod_drafts(
                credentials, pod_id=input_data.pod_id, **params
            )
            drafts = [d.model_dump() for d in response.drafts]

            yield "drafts", drafts
            yield "count", response.count
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailCreatePodInboxBlock(Block):
    """
    Create a new email inbox within a specific pod (customer workspace).

    The inbox is automatically scoped to the pod and inherits its
    isolation guarantees. If username/domain are not provided,
    AgentMail auto-generates a unique address.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        pod_id: str = SchemaField(description="Pod ID to create the inbox in")
        username: str = SchemaField(
            description="Local part of the email address (e.g. 'support'). Leave empty to auto-generate.",
            default="",
        )
        domain: str = SchemaField(
            description="Email domain (e.g. 'mydomain.com'). Defaults to agentmail.to if empty.",
            default="",
        )
        display_name: str = SchemaField(
            description="Friendly name shown in the 'From' field (e.g. 'Customer Support')",
            default="",
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(
            description="Unique identifier of the created inbox"
        )
        email_address: str = SchemaField(description="Full email address of the inbox")
        result: dict = SchemaField(
            description="Complete inbox object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="c6862373-1ac6-402e-89e6-7db1fea882af",
            description="Create a new email inbox within a pod. The inbox is scoped to the customer workspace.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "pod_id": "test-pod"},
            test_output=[
                ("inbox_id", "mock-inbox-id"),
                ("email_address", "mock-inbox-id"),
                ("result", dict),
            ],
            test_mock={
                "create_pod_inbox": lambda *a, **kw: type(
                    "Inbox",
                    (),
                    {
                        "inbox_id": "mock-inbox-id",
                        "model_dump": lambda self: {"inbox_id": "mock-inbox-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def create_pod_inbox(credentials: APIKeyCredentials, pod_id: str, **params):
        client = _client(credentials)
        return await client.pods.inboxes.create(pod_id=pod_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {}
            if input_data.username:
                params["username"] = input_data.username
            if input_data.domain:
                params["domain"] = input_data.domain
            if input_data.display_name:
                params["display_name"] = input_data.display_name

            inbox = await self.create_pod_inbox(
                credentials, pod_id=input_data.pod_id, **params
            )
            result = inbox.model_dump()

            yield "inbox_id", inbox.inbox_id
            yield "email_address", inbox.inbox_id
            yield "result", result
        except Exception as e:
            yield "error", str(e)
