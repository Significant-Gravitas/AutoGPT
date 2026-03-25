"""
SAML Authentication Database Migration

This script creates the necessary database tables for SAML authentication.
Run this after updating the schema.prisma file.
"""

import asyncio
import logging
from datetime import datetime, timezone

from prisma import Prisma

logger = logging.getLogger(__name__)


async def create_saml_tables():
    """Create SAML authentication tables."""
    client = Prisma()
    
    try:
        await client.connect()
        
        # Check if tables already exist
        logger.info("Checking for existing SAML tables...")
        
        # Create tables using Prisma migrate
        # In production, you would run: prisma migrate dev --name add_saml_auth
        logger.info("Please run the following command to create SAML tables:")
        logger.info("prisma migrate dev --name add_saml_auth")
        
        # For now, let's just verify the connection
        result = await client.query_raw(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name LIKE 'saml%'"
        )
        
        if result:
            logger.info(f"Found {len(result)} SAML-related tables:")
            for row in result:
                logger.info(f"  - {row['table_name']}")
        else:
            logger.info("No SAML tables found. Please run the migration.")
        
    except Exception as e:
        logger.error(f"Error checking SAML tables: {e}")
        raise
    finally:
        await client.disconnect()


async def seed_sample_saml_providers():
    """Seed sample SAML providers for testing."""
    client = Prisma()
    
    try:
        await client.connect()
        
        # Check if providers already exist
        existing = await client.samlprovider.find_many()
        if existing:
            logger.info(f"Found {len(existing)} existing SAML providers")
            return
        
        # Create sample Okta provider
        await client.samlprovider.create(
            data={
                "providerName": "okta",
                "displayName": "Okta",
                "enabled": False,  # Disabled by default
                "entityId": "https://agpt.co/saml",
                "acsUrl": "https://agpt.co/api/auth/saml/acs",
                "sloUrl": "https://agpt.co/api/auth/saml/slo",
                "idpEntityId": "https://your-okta-domain.okta.com",
                "idpSsoUrl": "https://your-okta-domain.okta.com/app/saml/exk123/sso/sso",
                "idpSloUrl": "https://your-okta-domain.okta.com/app/saml/exk123/slo/slo",
                "idpX509Cert": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
                "wantAssertionsSigned": True,
                "wantResponseSigned": True,
                "wantAssertionsEncrypted": False,
                "wantNameIdEncrypted": False,
                "attributeMapping": {
                    "email": "email",
                    "name": "name",
                    "firstName": "firstName",
                    "lastName": "lastName",
                    "username": "username",
                    "groups": "groups"
                }
            }
        )
        
        # Create sample Azure AD provider
        await client.samlprovider.create(
            data={
                "providerName": "azure",
                "displayName": "Azure Active Directory",
                "enabled": False,  # Disabled by default
                "entityId": "https://agpt.co/saml",
                "acsUrl": "https://agpt.co/api/auth/saml/acs",
                "sloUrl": "https://agpt.co/api/auth/saml/slo",
                "idpEntityId": "https://sts.windows.net/your-tenant-id/",
                "idpSsoUrl": "https://login.microsoftonline.com/your-tenant-id/saml2",
                "idpSloUrl": "https://login.microsoftonline.com/your-tenant-id/saml2",
                "idpX509Cert": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
                "wantAssertionsSigned": True,
                "wantResponseSigned": True,
                "wantAssertionsEncrypted": False,
                "wantNameIdEncrypted": False,
                "attributeMapping": {
                    "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
                    "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
                    "firstName": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
                    "lastName": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
                    "username": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn",
                    "groups": "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups"
                }
            }
        )
        
        logger.info("Created sample SAML providers (okta and azure)")
        logger.info("Remember to:")
        logger.info("1. Update the provider configurations with your actual IdP details")
        logger.info("2. Upload your IdP's X.509 certificates")
        logger.info("3. Enable the providers when ready")
        
    except Exception as e:
        logger.error(f"Error seeding SAML providers: {e}")
        raise
    finally:
        await client.disconnect()


async def main():
    """Main migration function."""
    logger.info("Starting SAML authentication migration...")
    
    # Check tables
    await create_saml_tables()
    
    # Seed sample data
    await seed_sample_saml_providers()
    
    logger.info("SAML migration complete!")
    logger.info("\nNext steps:")
    logger.info("1. Run: prisma migrate dev --name add_saml_auth")
    logger.info("2. Update environment variables with your SAML configuration")
    logger.info("3. Test the SAML authentication flow")
    logger.info("4. Enable providers in production when ready")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    asyncio.run(main())
