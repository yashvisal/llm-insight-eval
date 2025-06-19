"""
Test to understand E2B filesystem API
"""
from e2b_code_interpreter import Sandbox

def test_e2b_filesystem():
    """Test E2B filesystem operations"""
    print("Testing E2B filesystem...")
    
    with Sandbox() as sandbox:
        # Check the files attribute
        print("Files object attributes:", dir(sandbox.files))
        
        # Test file operations using the files attribute
        try:
            # Try to write a file using the files object
            sandbox.files.write("test.txt", "Hello World")
            print("✅ sandbox.files.write method works")
        except Exception as e:
            print(f"❌ sandbox.files.write failed: {e}")
        
        try:
            # Try to upload a file
            sandbox.files.upload("test2.txt", b"Hello World")
            print("✅ sandbox.files.upload method works")
        except Exception as e:
            print(f"❌ sandbox.files.upload failed: {e}")
        
        try:
            # Try to create a file
            sandbox.files.create("test3.txt", "Hello World")
            print("✅ sandbox.files.create method works")
        except Exception as e:
            print(f"❌ sandbox.files.create failed: {e}")

if __name__ == "__main__":
    test_e2b_filesystem() 