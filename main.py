import os
import sys
from src.field_identification import main

if __name__ == "__main__":
    # Ensure current directory is in path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
