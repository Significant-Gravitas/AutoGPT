# Security Policy

 - [**Using AutoGPT Securely**](#using-AutoGPT-securely)
   - [Restrict Workspace](#restrict-workspace)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Untrusted environments or networks](#untrusted-environments-or-networks)
   - [Multi-Tenant environments](#multi-tenant-environments)
 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)

## Using AutoGPT Securely

### Restrict Workspace

Since agents can read and write files, it is important to keep them restricted to a specific workspace. This happens by default *unless* RESTRICT_TO_WORKSPACE is set to False.

Disabling RESTRICT_TO_WORKSPACE can increase security risks. However, if you still need to disable it, consider running AutoGPT inside a [sandbox](https://developers.google.com/code-sandboxing), to mitigate some of these risks.

### Untrusted inputs

When handling untrusted inputs, it's crucial to isolate the execution and carefully pre-process inputs to mitigate script injection risks.

For maximum security when handling untrusted inputs, you may need to employ the following:

* Sandboxing: Isolate the process.
* Updates: Keep your libraries (including AutoGPT) updated with the latest security patches.
* Input Sanitation: Before feeding data to the model, sanitize inputs rigorously. This involves techniques such as:
    * Validation: Enforce strict rules on allowed characters and data types.
    * Filtering: Remove potentially malicious scripts or code fragments.
    * Encoding: Convert special characters into safe representations.
    * Verification: Run tooling that identifies potential script injections (e.g. [models that detect prompt injection attempts](https://python.langchain.com/docs/guides/safety/hugging_face_prompt_injection)). 

### Data privacy

To protect sensitive data from potential leaks or unauthorized access, it is crucial to sandbox the agent execution. This means running it in a secure, isolated environment, which helps mitigate many attack vectors.

### Untrusted environments or networks

Since AutoGPT performs network calls to the OpenAI API, it is important to always run it with trusted environments and networks. Running it on untrusted environments can expose your API KEY to attackers.
Additionally, running it on an untrusted network can expose your data to potential network attacks. 

However, even when running on trusted networks, it is important to always encrypt sensitive data while sending it over the network.

### Multi-Tenant environments

If you intend to run multiple AutoGPT brains in parallel, it is your responsibility to ensure the models do not interact or access each other's data.

The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.

- Tenant Isolation: you must make sure that the tenants run separately to prevent unwanted access to the data from other tenants. Keeping model network traffic separate is also important because you not only prevent unauthorized access to data, but also prevent malicious users or tenants sending prompts to execute under another tenant’s identity.

- Resource Allocation: a denial of service caused by one tenant can affect the overall system health. Implement safeguards like rate limits, access controls, and health monitoring.

- Data Sharing: in a multi-tenant design with data sharing, ensure tenants and users understand the security risks and sandbox agent execution to mitigate risks.

- Hardware Attacks: the hardware (GPUs or TPUs) can also be attacked. [Research](https://scholar.google.com/scholar?q=gpu+side+channel) has shown that side channel attacks on GPUs are possible, which can make data leak from other brains or processes running on the same system at the same time.

## Reporting a Vulnerability

Beware that none of the topics under [Using AutoGPT Securely](#using-AutoGPT-securely) are considered vulnerabilities on AutoGPT.

However, If you have discovered a security vulnerability in this project, please report it privately. **Do not disclose it as a public issue.** This gives us time to work with you to fix the issue before public exposure, reducing the chance that the exploit will be used before a patch is released.

Please disclose it as a private [security advisory](https://github.com/Significant-Gravitas/AutoGPT/security/advisories/new).

A team of volunteers on a reasonable-effort basis maintains this project. As such, please give us at least 90 days to work on a fix before public exposure.
