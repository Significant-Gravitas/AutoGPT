#!/usr/bin/env python3
"""
Example script demonstrating GCS file storage functionality.

This script shows how to use the new GCS file storage blocks to:
1. Store files permanently in Google Cloud Storage
2. Retrieve files using presigned URLs
3. Clean up expired files

Prerequisites:
- Set MEDIA_GCS_BUCKET_NAME environment variable
- Configure Google Cloud authentication (ADC or service account)
- Install required dependencies: google-cloud-storage
"""

import asyncio
import os
import tempfile
from datetime import datetime

from backend.blocks.gcs_file_store import GCSFileStoreBlock
from backend.blocks.gcs_file_retrieve import GCSFileRetrieveBlock
# No manual cleanup needed - GCS bucket lifecycle policies handle automatic deletion


async def demonstrate_gcs_file_storage():
    """Demonstrate the GCS file storage functionality."""
    
    print("🚀 AutoGPT GCS File Storage Demo")
    print("=" * 50)
    
    # Check configuration
    bucket_name = os.getenv('MEDIA_GCS_BUCKET_NAME')
    if not bucket_name:
        print("❌ Error: MEDIA_GCS_BUCKET_NAME environment variable not set")
        print("   Please set it to your GCS bucket name")
        return
    
    print(f"📦 Using GCS bucket: {bucket_name}")
    print()
    
    try:
        # Initialize blocks
        store_block = GCSFileStoreBlock()
        retrieve_block = GCSFileRetrieveBlock()
        
        print("✅ GCS blocks initialized successfully")
        print()
        
        # Example 1: Store a data URI
        print("📄 Example 1: Storing a data URI")
        print("-" * 30)
        
        data_uri = "data:text/plain;base64,SGVsbG8gZnJvbSBBdXRvR1BUIEdDUyBGaWxlIFN0b3JhZ2Uh"  # "Hello from AutoGPT GCS File Storage!"
        
        store_input = store_block.Input(
            file_in=data_uri,
            custom_path="examples/hello",
            expiration_hours=2  # 2 hours for demo
        )
        
        store_result = await store_block.run(store_input)
        print(f"✅ File stored successfully!")
        print(f"   📍 File path: {store_result.data['file_path']}")
        print(f"   🔗 File URL: {store_result.data['file_url']}")
        print(f"   ⏰ Expires at: {store_result.data['expiration_time']}")
        print()
        
        # Example 2: Retrieve the stored file
        print("📥 Example 2: Retrieving the stored file")
        print("-" * 35)
        
        retrieve_input = retrieve_block.Input(
            file_path=store_result.data['file_path'],
            access_duration_minutes=30,
            action="GET"
        )
        
        retrieve_result = await retrieve_block.run(retrieve_input)
        print(f"✅ File retrieved successfully!")
        print(f"   📂 File exists: {retrieve_result.data['file_exists']}")
        print(f"   📏 File size: {retrieve_result.data['file_size']} bytes")
        print(f"   🏷️  File type: {retrieve_result.data['file_type']}")
        print(f"   🔐 Presigned URL: {retrieve_result.data['presigned_url'][:50]}...")
        print(f"   ⏰ URL expires at: {retrieve_result.data['expires_at']}")
        print()
        
        # Example 3: Store a local file
        print("📁 Example 3: Storing a local file")
        print("-" * 32)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            import json
            sample_data = {
                "message": "This is a sample JSON file",
                "timestamp": datetime.utcnow().isoformat(),
                "demo": True
            }
            json.dump(sample_data, temp_file, indent=2)
            temp_file_path = temp_file.name
        
        try:
            store_input_2 = store_block.Input(
                file_in=temp_file_path,
                custom_path="examples/data",
                expiration_hours=1  # 1 hour for demo
            )
            
            store_result_2 = await store_block.run(store_input_2)
            print(f"✅ JSON file stored successfully!")
            print(f"   📍 File path: {store_result_2.data['file_path']}")
            print(f"   🔗 File URL: {store_result_2.data['file_url']}")
            print()
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
        
        # Example 4: Store from URL (if internet available)
        print("🌐 Example 4: Storing from URL")
        print("-" * 28)
        
        try:
            # Use a small public file for demo
            url = "https://httpbin.org/json"
            
            store_input_3 = store_block.Input(
                file_in=url,
                custom_path="examples/downloaded",
                expiration_hours=1
            )
            
            store_result_3 = await store_block.run(store_input_3)
            print(f"✅ URL content stored successfully!")
            print(f"   📍 File path: {store_result_3.data['file_path']}")
            print(f"   🔗 File URL: {store_result_3.data['file_url']}")
            print()
            
        except Exception as e:
            print(f"⚠️  URL storage failed (network may be unavailable): {e}")
            print()
        
        # Example 5: File lifecycle information
        print("♻️  Example 5: File lifecycle management")
        print("-" * 37)
        
        print("✅ Files are automatically deleted after 2 days by GCS lifecycle policies!")
        print("   📊 No manual cleanup required")
        print("   ♻️  Bucket lifecycle rules handle automatic deletion")
        print("   🕐 Files in 'autogpt-temp/' prefix expire after 2 days")
        print()
        
        print("🎉 GCS File Storage Demo completed successfully!")
        print()
        print("💡 Next steps:")
        print("   1. Use the file URLs in your agents instead of base64 data")
        print("   2. Set up GCS bucket lifecycle policies for automatic cleanup") 
        print("   3. Monitor storage usage and costs")
        print("   4. Configure appropriate expiration times for your use case")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Please check your GCS configuration and authentication")


def demonstrate_lifecycle_policies():
    """Demonstrate GCS lifecycle policy information."""
    
    print("\n♻️  GCS Lifecycle Management Info")
    print("=" * 40)
    
    print("🔧 GCS Bucket Lifecycle Policies:")
    print("   • Files in 'autogpt-temp/' are automatically deleted after 2 days")
    print("   • No manual intervention required")
    print("   • Lifecycle rules are configured at the bucket level")
    print("   • Cost-effective storage management")
    print()
    print("📋 To set up lifecycle policies on your bucket:")
    print('   1. Create lifecycle.json with deletion rule')
    print('   2. Run: gsutil lifecycle set lifecycle.json gs://your-bucket')
    print('   3. Verify: gsutil lifecycle get gs://your-bucket')
    print()
    print("✅ Lifecycle policy demo completed!")


if __name__ == "__main__":
    print("AutoGPT GCS File Storage Example")
    print("Please ensure you have:")
    print("1. Set MEDIA_GCS_BUCKET_NAME environment variable")
    print("2. Configured Google Cloud authentication")
    print("3. Installed google-cloud-storage package")
    print()
    
    # Run the main demo
    asyncio.run(demonstrate_gcs_file_storage())
    
    # Show lifecycle policy information
    demonstrate_lifecycle_policies()