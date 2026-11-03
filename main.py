import os
from core.registry import OCRPipelineManager

def main():
    # Initialize pipeline manager
    manager = OCRPipelineManager()
    
    # Path to sample document (image or PDF)
    sample_file = "sample.pdf"
    
    if not os.path.exists(sample_file):
        print(f"[{sample_file}] not found. Please place a sample image or PDF in the root directory.")
        return

    # List all available engines
    print("Available Engines:", list(manager.engines.keys()))
    
    # Test a specific engine
    target_engine = "docling"
    print(f"\nRunning extraction with [{target_engine}]...")
    
    try:
        result = manager.run(target_engine, sample_file)
        print("\n--- Success ---")
        print(f"Engine: {result.engine_name}")
        print(f"Extracted Text Length: {len(result.text)} characters")
        print(f"Markdown Output Preview:\n{result.markdown[:300] if result.markdown else result.text[:300]}...")
    except Exception as e:
        print(f"Error running {target_engine}: {e}")

if __name__ == "__main__":
    main()