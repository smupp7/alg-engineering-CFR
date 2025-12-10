"""
Verify all dependencies are installed correctly
"""

def test_imports():
    """Test all required imports work"""
    
    print("Testing imports...\n")
    
    # Core scientific
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy failed: {e}")
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy failed: {e}")
    
    # Data handling
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas failed: {e}")
    
    # Visualization
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib failed: {e}")
    
    try:
        import seaborn
        print(f"✅ Seaborn {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn failed: {e}")
    
    # Parallelization
    try:
        import joblib
        print(f"✅ Joblib {joblib.__version__}")
    except ImportError as e:
        print(f"❌ Joblib failed: {e}")
    
    try:
        import psutil
        print(f"✅ psutil {psutil.__version__}")
    except ImportError as e:
        print(f"❌ psutil failed: {e}")
    
    # OpenSpiel (most important!)
    try:
        import pyspiel
        print(f"✅ OpenSpiel (pyspiel module)")
    except ImportError as e:
        print(f"❌ OpenSpiel failed: {e}")
        print("   If this failed, we'll use custom game implementations")
    
    print("\n" + "="*60)

def test_basic_functionality():
    """Test basic operations work"""
    
    print("\nTesting basic functionality...\n")
    
    # NumPy operations
    import numpy as np
    A = np.random.rand(100, 100)
    x = np.random.rand(100)
    y = A @ x
    print(f"✅ NumPy matrix multiply works")
    
    # Sparse operations
    from scipy.sparse import csr_matrix
    A_sparse = csr_matrix(A)
    y_sparse = A_sparse @ x
    print(f"✅ SciPy sparse operations work")
    
    # Parallel operations
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=2)(delayed(lambda x: x**2)(i) for i in range(10))
    print(f"✅ Joblib parallelization works")
    
    # CPU info
    import psutil
    import os
    print(f"✅ System info: {os.cpu_count()} CPU cores")
    
    print("\n" + "="*60)

def test_openspiel():
    """Test OpenSpiel specifically"""
    
    print("\nTesting OpenSpiel...\n")
    
    try:
        import pyspiel
        
        # Load Kuhn poker
        game = pyspiel.load_game("kuhn_poker")
        print(f"✅ Loaded Kuhn Poker")
        print(f"   Players: {game.num_players()}")
        print(f"   Max game length: {game.max_game_length()}")
        
        # Create initial state
        state = game.new_initial_state()
        print(f"✅ Created game state")
        
        # Try Leduc too
        game_leduc = pyspiel.load_game("leduc_poker")
        print(f"✅ Loaded Leduc Poker")
        
        print("\n✅✅✅ OpenSpiel is fully functional! ✅✅✅")
        return True
        
    except ImportError as e:
        print(f"❌ OpenSpiel not available: {e}")
        print("\n⚠️  Will need to use custom game implementations")
        return False
    except Exception as e:
        print(f"❌ OpenSpiel error: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("INSTALLATION VERIFICATION")
    print("="*60)
    
    test_imports()
    test_basic_functionality()
    openspiel_works = test_openspiel()
    
    print("\n" + "="*60)
    if openspiel_works:
        print("✅ ALL SYSTEMS GO! Ready to start implementing CFR")
        print("   Use OpenSpiel for game implementations")
    else:
        print("⚠️  OpenSpiel not available")
        print("   Will provide custom Kuhn/Leduc implementations")
    print("="*60)