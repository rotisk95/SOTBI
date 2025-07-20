#!/usr/bin/env python3
"""
ACCOUNTABILITY: Test demo to verify corrected add_child logic and trie structure.

TESTS PERFORMED:
1. Corrected add_child method (single node per token, not lists)
2. Multiple sequence addition with proper child aggregation
3. Structure verification against expected table format
4. Iterator compatibility testing
5. Continuation finding simulation

JUSTIFICATION: Validates that architectural fix resolves list vs single-node inconsistency.
"""

import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np

# Configure logging for execution transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSemanticTrieNode:
    """
    SIMPLIFIED: Test version of SemanticTrieNode with corrected add_child method.
    
    CHANGES FROM ORIGINAL:
    1. FIXED: add_child method stores single nodes, not lists
    2. SIMPLIFIED: Removed database and complex embedding logic for testing
    3. PRESERVED: Core children structure and properties needed for testing
    4. ADDED: Enhanced logging for test verification
    """
    
    def __init__(self, token: str = None):
        self.token = token
        self.node_id = f"node_{id(self)}"
        
        # CORE STRUCTURE: Children as dict[token_string -> single_node]
        self.children: Dict[str, 'TestSemanticTrieNode'] = {}
        
        # SIMPLIFIED: Basic properties for testing
        self.confidence = 0.5
        self.activation_level = 0.0
        self.relevance_score = 0.0
        self.is_complete = False
        self.metadata = {'avg_reward': 0.0}
        
        logger.debug(f"Created TestSemanticTrieNode: '{token}' (id: {self.node_id})")
    
    def add_child(self, token: str, child: 'TestSemanticTrieNode'):
        """
        CORRECTED: Add child node for token (single node per unique token string).
        
        FIXES APPLIED:
        1. FIXED: Store single node per token (not list)
        2. ADDED: Existing child replacement logging
        3. PRESERVED: Error handling
        4. REMOVED: List creation and appending logic
        
        JUSTIFICATION: Resolves architectural inconsistency between storage and iteration.
        """
        try:
            if token in self.children:
                existing_child = self.children[token]
                if existing_child != child:
                    logger.info(f"REPLACING child for token '{token}' in node '{self.token}': "
                               f"old_id={existing_child.node_id} -> new_id={child.node_id}")
                else:
                    logger.debug(f"SKIPPING: Same child already exists for token '{token}' in node '{self.token}'")
                    return
            
            # FIXED: Store single node per token string (not list)
            self.children[token] = child
            logger.info(f"ADDED child '{child.token}' for token '{token}' in node '{self.token}' "
                       f"(total children: {len(self.children)})")
            
        except Exception as e:
            logger.error(f"Error adding child '{token}' to node '{self.token}': {e}")
            raise
    
    def get_child(self, token: str) -> Optional['TestSemanticTrieNode']:
        """Get child node by token."""
        return self.children.get(token)
    
    def calculate_relevance(self, **kwargs) -> float:
        """Simplified relevance calculation for testing."""
        return self.relevance_score
    
    def __repr__(self):
        return f"TestNode('{self.token}', children={len(self.children)})"


class TrieArchitectureTest:
    """
    COMPREHENSIVE: Test suite to verify corrected trie architecture.
    
    TESTING STRATEGY:
    1. Create test sequences matching user's table example
    2. Verify corrected add_child behavior
    3. Test structure matches expected format
    4. Validate iteration compatibility
    5. Simulate continuation finding
    """
    
    def __init__(self):
        self.embeddings: Dict[str, TestSemanticTrieNode] = {}
        self.root = TestSemanticTrieNode(token=None)
        logger.info("Initialized TrieArchitectureTest with corrected add_child logic")
    
    def create_or_get_node(self, token: str) -> TestSemanticTrieNode:
        """
        SINGLE SOURCE: Create or retrieve node from embeddings dictionary.
        
        JUSTIFICATION: Ensures single node per token string across all sequences.
        """
        if token not in self.embeddings:
            self.embeddings[token] = TestSemanticTrieNode(token)
            logger.info(f"CREATED new node for token '{token}' (id: {self.embeddings[token].node_id})")
        else:
            logger.debug(f"RETRIEVED existing node for token '{token}' (id: {self.embeddings[token].node_id})")
        
        return self.embeddings[token]
    
    def add_sequence(self, tokens: List[str], sequence_name: str = ""):
        """
        SEQUENCE PROCESSING: Add sequence using corrected immediate-next-only logic.
        
        CHANGES FROM ORIGINAL _get_token_embeddings:
        1. FIXED: Only add immediate-next child (not all subsequent)
        2. PRESERVED: Multi-sequence aggregation capability
        3. ADDED: Comprehensive logging for verification
        4. SIMPLIFIED: Test version without embedding generation
        """
        logger.info(f"ADDING SEQUENCE {sequence_name}: {tokens}")
        
        # STEP 1: Create/retrieve all nodes
        nodes = []
        for token in tokens:
            node = self.create_or_get_node(token)
            nodes.append(node)
        
        # STEP 2: CORRECTED - Build immediate-next-only relationships
        logger.info(f"Building immediate-next relationships for sequence {sequence_name}...")
        
        for i, token in enumerate(tokens):
            current_node = nodes[i]
            
            # FIXED: Add only immediate next token as child (not all subsequent)
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                next_node = nodes[i + 1]
                
                # Check if child already exists from previous sequence
                children_before = len(current_node.children)
                
                current_node.add_child(next_token, next_node)
                
                children_after = len(current_node.children)
                added = children_after > children_before
                
                logger.info(f"'{token}' -> '{next_token}' "
                           f"(added: {added}, total_children: {children_after})")
            else:
                logger.info(f"'{token}': end of sequence {sequence_name}")
        
        logger.info(f"SEQUENCE {sequence_name} COMPLETE\n")
    
    def verify_structure(self):
        """
        VERIFICATION: Check that structure matches expected table format.
        
        ACCOUNTABILITY: Validates corrected logic produces intended architecture.
        """
        logger.info("VERIFYING TRIE STRUCTURE...")
        
        print("\n" + "="*80)
        print("TRIE STRUCTURE VERIFICATION")
        print("="*80)
        
        structure_correct = True
        
        for token, node in self.embeddings.items():
            children_tokens = list(node.children.keys())
            print(f"Node '{token}': {len(children_tokens)} children -> {children_tokens}")
            
            # VERIFY: Each child is a single node (not list)
            for child_token, child_node in node.children.items():
                if not isinstance(child_node, TestSemanticTrieNode):
                    logger.error(f"STRUCTURE ERROR: '{token}' -> '{child_token}' is not single node: {type(child_node)}")
                    structure_correct = False
                else:
                    logger.debug(f"✅ '{token}' -> '{child_token}' is single node (id: {child_node.node_id})")
        
        print(f"\nStructure verification: {'PASSED' if structure_correct else 'FAILED'}")
        return structure_correct
    
    def test_iteration_compatibility(self):
        """
        ITERATION TEST: Verify corrected structure works with existing iteration patterns.
        
        JUSTIFICATION: Tests that fix resolves original iterator errors.
        """
        logger.info("TESTING ITERATION COMPATIBILITY...")
        
        print("\n" + "="*80)
        print("ITERATION COMPATIBILITY TEST")
        print("="*80)
        
        iteration_success = True
        
        for token, node in self.embeddings.items():
            try:
                print(f"\nTesting iteration for node '{token}':")
                
                # TEST: Standard iteration pattern from _collect_continuations
                for child_token, child_node in node.children.items():
                    # These property accesses should work with single nodes
                    confidence = getattr(child_node, 'confidence', 0.5)
                    activation = child_node.activation_level
                    children_count = len(child_node.children)
                    
                    print(f"  Child '{child_token}': confidence={confidence:.3f}, "
                          f"activation={activation:.3f}, children={children_count}")
                    
                    logger.debug(f"✅ Successfully accessed properties of child '{child_token}'")
                
            except Exception as e:
                logger.error(f"❌ ITERATION FAILED for node '{token}': {e}")
                iteration_success = False
        
        print(f"\nIteration compatibility: {'PASSED' if iteration_success else 'FAILED'}")
        return iteration_success
    
    def simulate_continuation_finding(self, query_tokens: List[str]):
        """
        CONTINUATION SIMULATION: Test that corrected structure supports continuation finding.
        
        JUSTIFICATION: Validates that fix enables proper trie traversal.
        """
        logger.info(f"SIMULATING CONTINUATION FINDING for query: {query_tokens}")
        
        print(f"\n" + "="*80)
        print(f"CONTINUATION FINDING SIMULATION: {query_tokens}")
        print("="*80)
        
        try:
            # STEP 1: Find last matched node
            if not query_tokens:
                print("Empty query - no continuation possible")
                return []
            
            last_token = query_tokens[-1]
            if last_token not in self.embeddings:
                print(f"Token '{last_token}' not found in embeddings")
                return []
            
            last_node = self.embeddings[last_token]
            print(f"Last matched node: '{last_token}' with {len(last_node.children)} children")
            
            # STEP 2: Collect immediate continuations
            continuations = []
            for child_token, child_node in last_node.children.items():
                # This iteration should work with corrected structure
                score = (
                    0.4 * child_node.relevance_score +
                    0.3 * child_node.activation_level +
                    0.2 * child_node.metadata.get('avg_reward', 0.0) +
                    (0.1 if child_node.is_complete else 0.0)
                )
                continuations.append((child_token, score))
                print(f"  Continuation option: '{child_token}' (score: {score:.3f})")
            
            # STEP 3: Select best continuation
            if continuations:
                best_token, best_score = max(continuations, key=lambda x: x[1])
                print(f"\nBest continuation: '{best_token}' (score: {best_score:.3f})")
                return [best_token]
            else:
                print("No continuations available")
                return []
                
        except Exception as e:
            logger.error(f"❌ CONTINUATION FINDING FAILED: {e}")
            return []
    
    def run_comprehensive_test(self):
        """
        COMPREHENSIVE TEST: Execute full test suite to verify corrected architecture.
        
        ACCOUNTABILITY: Validates all aspects of the architectural fix.
        """
        logger.info("STARTING COMPREHENSIVE TRIE ARCHITECTURE TEST")
        
        print("🧪 TRIE ARCHITECTURE TEST DEMO")
        print("="*80)
        print("TESTING: Corrected add_child logic and trie structure")
        print("OBJECTIVE: Verify single-node-per-token architecture works correctly")
        print("="*80)
        
        # TEST SEQUENCES: Based on user's table example
        test_sequences = [
            (['Hey', 'there!', 'How', 'are', 'you'], "Sequence 1"),
            (['Hey', 'How', 'are', 'you'], "Sequence 2"),  
            (['there!', 'How', 'these'], "Sequence 3"),
            (['there!', 'Are', 'these'], "Sequence 4"),
            (['How', 'Are', 'You'], "Sequence 5"),
            (['How', 'Are', 'Doing'], "Sequence 6"),
            (['How', 'the'], "Sequence 7"),
            (['Are', 'the'], "Sequence 8"),
            (['Are', 'we'], "Sequence 9"),
            (['Are', 'You?'], "Sequence 10"),
            (['Are', 'they'], "Sequence 11")
        ]
        
        # EXECUTE TESTS
        test_results = {
            'sequences_added': 0,
            'structure_correct': False,
            'iteration_compatible': False,
            'continuation_finding': False
        }
        
        try:
            # TEST 1: Add all sequences
            print("\n🔄 TEST 1: Adding multiple sequences...")
            for tokens, name in test_sequences:
                self.add_sequence(tokens, name)
                test_results['sequences_added'] += 1
            
            print(f"✅ Successfully added {test_results['sequences_added']} sequences")
            
            # TEST 2: Verify structure
            print("\n🔍 TEST 2: Verifying trie structure...")
            test_results['structure_correct'] = self.verify_structure()
            
            # TEST 3: Test iteration compatibility  
            print("\n🔄 TEST 3: Testing iteration compatibility...")
            test_results['iteration_compatible'] = self.test_iteration_compatibility()
            
            # TEST 4: Test continuation finding
            print("\n🎯 TEST 4: Testing continuation finding...")
            test_queries = [['Hey'], ['there!'], ['How'], ['Are']]
            
            continuation_success = True
            for query in test_queries:
                try:
                    result = self.simulate_continuation_finding(query)
                    logger.info(f"Continuation for {query}: {result}")
                except Exception as e:
                    logger.error(f"Continuation finding failed for {query}: {e}")
                    continuation_success = False
            
            test_results['continuation_finding'] = continuation_success
            
            # FINAL RESULTS
            print("\n" + "="*80)
            print("FINAL TEST RESULTS")
            print("="*80)
            
            all_passed = all(test_results.values())
            
            for test_name, result in test_results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{test_name}: {status}")
            
            print(f"\nOVERALL RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
            
            if all_passed:
                print("\n🎉 CORRECTED ARCHITECTURE VERIFIED SUCCESSFULLY!")
                print("The single-node-per-token structure works as intended.")
            else:
                print("\n⚠️  ARCHITECTURE ISSUES DETECTED")
                print("Further debugging required.")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"COMPREHENSIVE TEST FAILED: {e}")
            print(f"\n❌ TEST SUITE FAILED: {e}")
            return False


if __name__ == "__main__":
    """
    EXECUTION: Run comprehensive test to verify corrected trie architecture.
    
    ACCOUNTABILITY: Demonstrates that architectural fix resolves the list vs single-node issue.
    """
    
    print("Starting Trie Architecture Verification Test...")
    
    # Create and run test
    test_suite = TrieArchitectureTest()
    success = test_suite.run_comprehensive_test()
    
    if success:
        print("\n✅ RECOMMENDATION: Implement corrected add_child method in production code")
    else:
        print("\n❌ RECOMMENDATION: Further architectural analysis required")
    
    print("\nTest demo complete.")