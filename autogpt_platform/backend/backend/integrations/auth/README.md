# SAML 2.0 Authentication Integration

This module provides SAML 2.0 authentication support for AutoGPT, enabling enterprise single sign-on (SSO) with identity providers like Okta, Azure AD, ADFS, and other SAML 2.0 compliant IdPs.

## Features

- **Multiple Provider Support**: Configure multiple SAML identity providers
- **Security**: Signed/encrypted assertions, certificate-based trust
- **User Provisioning**: Automatic user creation and attribute mapping
- **Session Management**: SAML session tracking and logout support
- **Flexible Configuration**: Database-driven provider configuration
- **Audit Trail**: Request tracking for security auditing

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   SAML API       │────▶│  SAML Manager    │
│                 │     │                  │     │                 │
│ - Login Button  │     │ - /login         │     │ - python3-saml2  │
│ - Redirect      │     │ - /acs           │     │ - Config Mgmt   │
│                 │     │ - /logout        │     │ - Response Proc │
└─────────────────┘     │ - /metadata      │     └─────────────────┘
                        └──────────────────┘              │
                                 │                        │
                                 ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   Data Layer     │     │   Database      │
                        │                  │     │                 │
                        │ - ProviderData   │     │ - SAMLProvider  │
                        │ - UserData       │     │ - SAMLUser      │
                        │ - AuthService    │     │ - SAMLAuthReq   │
                        └──────────────────┘     └─────────────────┘
```

## Installation

1. **Install Dependencies**:
   ```bash
   # python3-saml2 is already added to pyproject.toml
   poetry install
   ```

2. **Run Database Migration**:
   ```bash
   prisma migrate dev --name add_saml_auth
   ```

3. **Seed Sample Providers** (optional):
   ```bash
   python backend/integrations/auth/migrate_saml.py
   ```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# SAML Service Provider Configuration
SAML_SP_ENTITY_ID=https://your-domain.com/saml
SAML_ACS_URL=https://your-domain.com/api/auth/saml/acs
SAML_SLO_URL=https://your-domain.com/api/auth/saml/slo

# Okta Configuration (example)
SAML_OKTA_ENABLED=true
SAML_OKTA_ENTITY_ID=https://your-okta-domain.okta.com
SAML_OKTA_SSO_URL=https://your-okta-domain.okta.com/app/saml/exk123/sso/sso
SAML_OKTA_SLO_URL=https://your-okta-domain.okta.com/app/saml/exk123/slo/slo
SAML_OKTA_X509_CERT=-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----

# Azure AD Configuration (example)
SAML_AZURE_ENABLED=true
SAML_AZURE_ENTITY_ID=https://sts.windows.net/your-tenant-id/
SAML_AZURE_SSO_URL=https://login.microsoftonline.com/your-tenant-id/saml2
SAML_AZURE_SLO_URL=https://login.microsoftonline.com/your-tenant-id/saml2
SAML_AZURE_X509_CERT=-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----
```

### Provider Configuration

Providers can be configured via the API or directly in the database:

```python
# Example: Configure Okta provider
{
    "providerName": "okta",
    "displayName": "Okta",
    "enabled": true,
    "entityId": "https://your-domain.com/saml",
    "acsUrl": "https://your-domain.com/api/auth/saml/acs",
    "sloUrl": "https://your-domain.com/api/auth/saml/slo",
    "idpEntityId": "https://your-okta-domain.okta.com",
    "idpSsoUrl": "https://your-okta-domain.okta.com/app/saml/exk123/sso/sso",
    "idpSloUrl": "https://your-okta-domain.okta.com/app/saml/exk123/slo/slo",
    "idpX509Cert": "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
    "attributeMapping": {
        "email": "email",
        "name": "name",
        "firstName": "firstName",
        "lastName": "lastName",
        "groups": "groups"
    }
}
```

## API Endpoints

### Authentication Flow

1. **Initiate Login**:
   ```http
   POST /api/auth/saml/login
   {
       "provider_name": "okta",
       "relay_state": "optional-state"
   }
   ```

2. **Login via GET** (for redirects):
   ```http
   GET /api/auth/saml/login/okta?relay_state=optional-state
   ```

3. **Assertion Consumer Service**:
   ```http
   POST /api/auth/saml/acs
   Content-Type: application/x-www-form-urlencoded
   
   SAMLResponse=<base64-encoded-response>&RelayState=<state>
   ```

4. **Initiate Logout**:
   ```http
   GET /api/auth/saml/logout/okta
   Authorization: Bearer <user-token>
   ```

5. **Single Logout Service**:
   ```http
   POST /api/auth/saml/slo
   Content-Type: application/x-www-form-urlencoded
   
   SAMLResponse=<base64-encoded-response>
   ```

### Provider Management

6. **List Providers**:
   ```http
   GET /api/auth/saml/providers?enabled_only=true
   ```

7. **Get Metadata**:
   ```http
   GET /api/auth/saml/metadata/okta
   ```

8. **User Providers**:
   ```http
   GET /api/auth/saml/user/providers
   Authorization: Bearer <user-token>
   ```

### Admin Endpoints

9. **Create Provider** (Admin only):
   ```http
   POST /api/auth/saml/admin/providers
   Authorization: Bearer <admin-token>
   
   {
       "providerName": "new-provider",
       "displayName": "New Provider",
       "enabled": true,
       // ... other config
   }
   ```

## Integration with IdPs

### Okta Setup

1. In Okta Admin Console, go to **Applications** → **Applications**
2. Click **Create App Integration** → **SAML 2.0**
3. Configure:
   - **Single Sign On URL**: `https://your-domain.com/api/auth/saml/acs`
   - **Audience URI (SP Entity ID)**: `https://your-domain.com/saml`
   - **Default Relay State**: Optional
4. In **Settings** → **Attributes**, map attributes:
   - `email` → `user.email`
   - `name` → `user.displayName`
   - `firstName` → `user.firstName`
   - `lastName` → `user.lastName`
5. Download the **IdP metadata** or copy the certificate
6. Update your AutoGPT configuration with the IdP details

### Azure AD Setup

1. In Azure Portal, go to **Azure Active Directory** → **Enterprise applications**
2. Click **New application** → **Create your own application**
3. Select **Integrate any other application you don't find in the gallery**
4. In **Single sign-on**, select **SAML**
5. Configure:
   - **Identifier (Entity ID)**: `https://your-domain.com/saml`
   - **Reply URL (Assertion Consumer Service URL)**: `https://your-domain.com/api/auth/saml/acs`
6. In **Attributes & Claims**, configure claims:
   - Email address: `user.email`
   - Name: `user.displayName`
   - Given Name: `user.givenname`
   - Surname: `user.surname`
   - UPN: `user.userprincipalname`
7. Download the **Federation Metadata XML** or copy the certificate
8. Update your AutoGPT configuration

## Usage Examples

### Frontend Integration

```javascript
// Initiate SAML login
async function loginWithSAML(provider) {
    try {
        const response = await fetch('/api/auth/saml/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider_name: provider,
                relay_state: window.location.href
            })
        });
        
        const { auth_url } = await response.json();
        window.location.href = auth_url;
    } catch (error) {
        console.error('SAML login failed:', error);
    }
}

// Login button
<button onClick={() => loginWithSAML('okta')}>
    Login with Okta
</button>
```

### Backend Integration

```python
from backend.integrations.auth.saml import get_saml_manager

# Get SAML manager
manager = get_saml_manager()

# Check if provider is configured
if "okta" in manager.list_providers():
    # Show Okta login option
    pass
```

## Testing

### Run Tests

```bash
# Run SAML tests
python -m pytest backend/integrations/auth/saml_test.py -v

# Run with coverage
python -m pytest backend/integrations/auth/saml_test.py --cov=backend.integrations.auth
```

### Test with Test IdP

For testing, you can use a test SAML IdP like [SAMLTest.id](https://samltest.id/):

1. Configure test provider:
   ```python
   test_config = SAMLProviderConfig(
       provider_name="samltest",
       entity_id="https://your-domain.com/saml",
       acs_url="https://your-domain.com/api/auth/saml/acs",
       idp_entity_id="https://samltest.id/saml/idp",
       idp_sso_url="https://samltest.id/idp/profile/SAML2/Redirect/SSO",
       idp_x509_cert="-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----"
   )
   ```

2. Test the flow using the test IdP

## Security Considerations

1. **Certificate Validation**: Always validate IdP certificates
2. **Signed Assertions**: Enable `want_assertions_signed=True`
3. **Encryption**: Consider enabling assertion encryption for sensitive data
4. **Replay Protection**: Implement request ID tracking (included)
5. **Logout**: Implement proper Single Logout to terminate sessions
6. **Rate Limiting**: Apply rate limiting to SAML endpoints
7. **Audit Logging**: All SAML operations are logged for security auditing

## Troubleshooting

### Common Issues

1. **"Provider not found"**:
   - Check if provider is registered and enabled
   - Verify provider name matches exactly

2. **"Invalid SAML response"**:
   - Check if response is properly base64 encoded
   - Verify IdP certificate is correct
   - Check clock skew between SP and IdP

3. **"No NameID found"**:
   - Ensure IdP is sending NameID in the response
   - Check NameID format configuration

4. **"Certificate validation failed"**:
   - Verify certificate format (PEM)
   - Check for extra whitespace or line breaks
   - Ensure certificate is not expired

### Debug Logging

Enable debug logging for SAML:

```python
import logging
logging.getLogger('backend.integrations.auth.saml').setLevel(logging.DEBUG)
```

## Production Checklist

- [ ] Configure production URLs (HTTPS required)
- [ ] Upload valid IdP certificates
- [ ] Enable signed assertions
- [ ] Configure proper attribute mapping
- [ ] Test login/logout flows
- [ ] Set up monitoring for SAML endpoints
- [ ] Configure backup IdP if needed
- [ ] Document SAML configuration for your team
- [ ] Test with actual IdP accounts
- [ ] Verify user provisioning works correctly

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify your IdP configuration
3. Test with a known working provider first
4. Review the SAML specification if needed

## References

- [SAML 2.0 Specification](https://docs.oasis-open.org/security/saml/v2.0/)
- [python3-saml2 Documentation](https://github.com/IdentityPython/python3-saml2)
- [Okta SAML Setup Guide](https://developer.okta.com/docs/guides/saml-sso/)
- [Azure AD SAML Setup Guide](https://docs.microsoft.com/en-us/azure/active-directory/develop/saml-sso-service-provider)
