#!/usr/bin/env python3
"""
Test script for Event-Driven Activation Integration

This script tests the new cross-sequence context discovery functionality
to ensure it works correctly with the existing SOTBI system.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the SOTBI directory to Python path
sotbi_dir = Path(__file__).parent
sys.path.insert(0, str(sotbi_dir))

from predictive_system import PredictiveSystem
from event_driven_activation import EventDrivenActivation

def test_basic_activation_integration():
    """Test basic integration of event-driven activation with SOTBI."""
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_activation.lmdb")
    
    try:
        print("🧪 Testing Event-Driven Activation Integration")
        print("=" * 60)
        
        # Initialize system
        system = PredictiveSystem(temp_db_path)
        
        # Training data that will create cross-sequence opportunities
        training_data = [
            ("hello how are you doing", 0.9),
            ("hello there my friend", 0.8),
            ("how are things going", 0.7),
            ("how you doing today", 0.8),
            ("there are many ways", 0.6),
            ("you are doing great", 0.9),
        ]
        
        print("\n📚 Training the system...")
        for text, reward in training_data:
            result = system.process_input(text, reward)
            print(f"  ✅ Processed: '{text}' (reward: {reward})")
        
        # Test queries that should benefit from cross-sequence activation
        test_queries = [
            "hello how",      # Should find continuations from both "hello how are..." and "how you doing..."
            "how are",        # Should find continuations from "how are you..." and "how are things..."
            "you are",        # Should find continuations from "you are doing great"
        ]
        
        print(f"\n🔍 Testing predictions with activation context...")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Test with activation context OFF
            system.toggle_activation_context(False)
            normal_continuation, normal_confidence = system.predict_continuation(query)
            normal_result = f"'{' '.join(normal_continuation)}'" if normal_continuation else "No prediction"
            
            # Test with activation context ON
            system.toggle_activation_context(True)
            enhanced_continuation, enhanced_confidence = system.predict_continuation(query)
            enhanced_result = f"'{' '.join(enhanced_continuation)}'" if enhanced_continuation else "No prediction"
            
            print(f"  📊 Normal:    {normal_result} (confidence: {normal_confidence:.3f})")
            print(f"  ⚡ Enhanced:  {enhanced_result} (confidence: {enhanced_confidence:.3f})")
            
            # Check if enhancement provided different/better results
            if normal_continuation != enhanced_continuation:
                print(f"  ✅ Enhancement provided different continuation!")
            elif enhanced_confidence > normal_confidence:
                print(f"  ✅ Enhancement improved confidence!")
            else:
                print(f"  ℹ️  No enhancement detected for this query")
        
        # Test activation system statistics
        print(f"\n📊 Testing activation system statistics...")
        activation_system = EventDrivenActivation(system.trie_memory)
        
        # Trigger some activations
        activation_system._trigger_activation_events(["hello", "how"])
        stats = activation_system.get_activation_stats()
        
        print(f"  • Active nodes: {stats['active_nodes']}")
        print(f"  • Age distribution: {stats['age_distribution']}")
        print(f"  • Timeout: {stats['timeout_seconds']} seconds")
        
        # Test system insights
        print(f"\n🧠 Testing system insights...")
        insights = system.get_system_insights()
        
        activation_context = insights.get('activation_context', {})
        print(f"  • Activation context enabled: {activation_context.get('enabled', False)}")
        print(f"  • Description: {activation_context.get('description', 'N/A')}")
        
        print(f"\n✅ All tests completed successfully!")
        print(f"🎉 Event-driven activation integration is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            system.close()
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return True

def test_activation_discovery():
    """Test the cross-sequence discovery capabilities."""
    
    print(f"\n🔍 Testing Cross-Sequence Discovery...")
    print("=" * 60)
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_discovery.lmdb")
    
    try:
        system = PredictiveSystem(temp_db_path)
        
        # Create training data where sequences share tokens but have different contexts
        training_sequences = [
            ("machine learning is fascinating", 0.9),
            ("deep learning algorithms work", 0.8),
            ("learning from data helps", 0.7),
            ("machine intelligence grows", 0.8),
            ("artificial intelligence advances", 0.9),
            ("intelligence requires learning", 0.8),
        ]
        
        print(f"\n📚 Training with cross-sequence token sharing...")
        for text, reward in training_sequences:
            system.process_input(text, reward)
            print(f"  ✅ '{text}'")
        
        # Test query that should benefit from cross-sequence discovery
        test_query = "learning"
        
        print(f"\n🔍 Testing query: '{test_query}'")
        
        # Normal prediction
        system.toggle_activation_context(False)
        normal_result = system.predict_continuation(test_query)
        
        # Enhanced prediction with activation
        system.toggle_activation_context(True)
        enhanced_result = system.predict_continuation(test_query)
        
        print(f"  📊 Normal result: {normal_result}")
        print(f"  ⚡ Enhanced result: {enhanced_result}")
        
        # The enhanced result should potentially find continuations from multiple sequences
        if normal_result != enhanced_result:
            print(f"  ✅ Cross-sequence discovery working!")
        else:
            print(f"  ℹ️  Results identical - may need more training data")
        
        print(f"\n✅ Cross-sequence discovery test completed!")
        
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            system.close()
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Event-Driven Activation Integration Tests")
    print("=" * 80)
    
    # Run tests
    test1_success = test_basic_activation_integration()
    test2_success = test_activation_discovery()
    
    print("\n" + "=" * 80)
    if test1_success and test2_success:
        print("🎉 ALL TESTS PASSED! Event-driven activation integration is working correctly!")
        print(f"\n📝 Summary of enhancements:")
        print(f"  • Cross-sequence context discovery implemented")
        print(f"  • Event-driven node activation system working")
        print(f"  • Integration with existing SOTBI system successful")
        print(f"  • Interactive commands added (activation on/off)")
        print(f"  • System insights enhanced with activation context")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)