"""
FIXED: Complete Enhanced Trie-Based Predictive Token Embedding System

CRITICAL FIXES APPLIED:
1. FIXED beam search to only use actual trie children as continuations (not random nodes)
2. REMOVED complex fallback logic that was creating fake continuations
3. PRESERVED all working simple prediction logic
4. RESTORED basic trie traversal principles

ACCOUNTABILITY: Only _collect_multi_source_candidates method significantly changed.
All other functionality preserved exactly as working.
"""

import sys
import gc
import os
from queue import Queue
import sys
import msgpack
import numpy as np
import logging
import lmdb
import pickle
import time
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import string
from collections import defaultdict, deque
from dataclasses import dataclass
import concurrent.futures
import numpy as np

from hf_dataset_integration import HuggingFaceDatasetIntegration
from predictive_system import PredictiveSystem

# Configure logging for execution transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ADDED: HuggingFace integration imports and dependencies
try:
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HUGGINGFACE_AVAILABLE = True
    logger.info("HuggingFace datasets library available")
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("HuggingFace datasets library not available - dataset training will be disabled")


class UserInteraction:
    """
    PRESERVED: Interactive user interface with menu system for the trie-based learning system.
    """
    
    def __init__(self, db_path: str = "./trie_memory.lmdb"):
        """Initialize the interactive system with trie-based memory."""
        try:
            self.system = PredictiveSystem(db_path)
            self.hf_integration = HuggingFaceDatasetIntegration(self.system.trie_memory) if HUGGINGFACE_AVAILABLE else None
            self.session_stats = {
                'interactions': 0,
                'predictions': 0,
                'dataset_samples': 0,
                'start_time': time.time()
            }
            logger.info("Initialized UserInteraction with trie-based system")
        except Exception as e:
            logger.error(f"Failed to initialize UserInteraction: {e}")
            raise

    def show_menu(self):
        """PRESERVED: Display interactive menu for user"""
        print("\n" + "="*60)
        print("🧠 RECURSIVE TOKEN WEAVER - HYBRID LEARNING SYSTEM WITH ACTIVATION")
        print("="*60)
        print("1. Interactive Learning Mode (Real-time prediction with rl feedback)") 
        print("2. Train on HuggingFace datasets")
        print("3. View learning statistics")
        print("4. Save/Load model")
        print("5. Exit")
        print("="*60)

    def run_progressive_token_prediction(self, text: str, verbose: bool = True) -> Dict[str, Any]:
        """
        PRESERVED: Progressive token prediction method for HuggingFace integration compatibility.
        """
        try:
            logger.info(f"Running progressive token prediction for text: '{text[:50]}...'")
            
            tokens = self.system._tokenize(text)
            if not tokens:
                logger.warning("No tokens found in text")
                return {'total_predictions': 0, 'average_reward': 0.0}
            
            total_predictions = 0
            total_reward = 0.0
            
            for i in range(1, len(tokens)):
                context = tokens[:i]
                actual_next = tokens[i]
                
                try:
                    predicted_tokens, confidence = self.system.predict_continuation(' '.join(context))
                    
                    reward = 0.0
                    if predicted_tokens and len(predicted_tokens) > 0:
                        if predicted_tokens[0] == actual_next:
                            reward = confidence * 1.0
                        else:
                            reward = confidence * 0.1
                    
                    self.system.process_input(' '.join(context + [actual_next]), reward)
                    
                    total_predictions += 1
                    total_reward += reward
                    
                    if verbose:
                        logger.debug(f"Prediction {i}: context='{' '.join(context)}' actual='{actual_next}' "
                                   f"predicted='{predicted_tokens[0] if predicted_tokens else 'None'}' reward={reward:.3f}")
                        
                except Exception as pred_error:
                    logger.warning(f"Prediction failed at position {i}: {pred_error}")
                    self.system.process_input(' '.join(context + [actual_next]), 0.1)
                    total_predictions += 1
                    total_reward += 0.1
            
            average_reward = total_reward / total_predictions if total_predictions > 0 else 0.0
            
            result = {
                'total_predictions': total_predictions,
                'total_reward': total_reward,
                'average_reward': average_reward
            }
            
            logger.info(f"Progressive prediction completed: {total_predictions} predictions, avg reward: {average_reward:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in progressive token prediction: {e}")
            return {'total_predictions': 0, 'average_reward': 0.0}

    def _handle_interactive_learning(self):
        """PRESERVED: Interactive learning mode with beam search options and detailed feedback"""
        print("\n🎓 INTERACTIVE LEARNING MODE")
        print("Features:")
        print("• Real-time prediction with RL feedback")
        print("• Multi-node beam search with activation/RL scoring")
        print("• Detailed scoring breakdown available")
        print("\nCommands:")
        print("• Type text for prediction")
        print("• 'beam on/off' - toggle beam search")
        print("• 'details' - show detailed scoring")
        print("• 'insights' - show system insights")
        print("• 'debug <query>' - debug trie structure for query")
        print("• 'exit' - return to menu")
        
        use_beam_search = False  # Default to simple prediction (working correctly)
        show_details = False
        
        try:
            while True:
                print(f"\n🔍 Mode: {'Beam Search' if use_beam_search else 'Simple'} | Details: {'ON' if show_details else 'OFF'}")
                print("🗣️  Enter text or command:")
                raw = sys.stdin.read()  # reads until EOF
                user_input = raw.strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'beam on':
                    use_beam_search = True
                    print("✅ Beam search enabled")
                    continue
                elif user_input.lower() == 'beam off':
                    use_beam_search = False
                    print("✅ Simple prediction enabled")
                    continue
                elif user_input.lower() == 'details':
                    show_details = not show_details
                    print(f"✅ Detailed scoring: {'ON' if show_details else 'OFF'}")
                    continue
                elif user_input.lower() == 'insights':
                    self._show_system_insights()
                    continue
                # In your _handle_interactive_learning method, add:
                elif user_input.startswith('debug '):
                    query = user_input[6:]  # Remove 'debug '
                    tokens = self.system._tokenize(query)
                    self.system.trie_memory.debug_trie_structure(tokens)
                    continue
                
                if not user_input:
                    print("Please enter some text.")
                    continue
                
                try:
                    result = self.system.process_input(user_input, 0.5)
                    print(f"✅ Processed: {len(result['tokens'])} tokens")
                    
                    if show_details and use_beam_search:
                        prediction_result = self.system.predict_with_detailed_scoring(user_input)
                        continuation = prediction_result['predicted_continuation']
                        confidence = prediction_result['confidence_score']
                        
                        if continuation:
                            prediction = ''.join(continuation)
                            print(f"🔮 BEAM PREDICTION: '{user_input} {prediction}'")
                            print(f"📊 Confidence: {confidence:.3f}")
                            
                            if 'beam_search_details' in prediction_result:
                                details = prediction_result['beam_search_details']
                                if details:
                                    last_step = details[-1]
                                    print(f"🔍 Generation: {len(details)} steps, {last_step['num_candidates']} final candidates")
                            
                            weights = prediction_result['scoring_weights']
                            print(f"⚖️  Scoring weights: Activation({weights['activation_weight']:.2f}) "
                                 f"RL({weights['rl_weight']:.2f}) Relevance({weights['relevance_weight']:.2f}) "
                                 f"Coherence({weights['coherence_weight']:.2f}) Completeness({weights['completeness_weight']:.2f})")
                        else:
                            print("🤔 No beam prediction available")
                            
                    else:
                        continuation, confidence = self.system.predict_continuation(
                            user_input, use_beam_search=use_beam_search
                        )
                        
                        if continuation:
                            prediction = ' '.join(continuation)
                            method = "BEAM" if use_beam_search else "SIMPLE"
                            print(f"🔮 {method} PREDICTION: '{user_input} {prediction}' (confidence: {confidence:.3f})")
                        else:
                            print("🤔 No prediction available yet - keep training!")
                    
                    if continuation:
                        feedback = input("Rate prediction (0-1, or press Enter for 0.5): ").strip()
                        try:
                            reward = float(feedback) if feedback else 0.5
                            reward = max(0.0, min(1.0, reward))
                        except ValueError:
                            reward = 0.5
                        
                        full_sequence = f"{user_input} {' '.join(continuation)}"
                        self.system.process_input(full_sequence, reward)
                        print(f"✅ Updated with reward: {reward} (Method: {'Beam' if use_beam_search else 'Simple'})")
                    
                    self.session_stats['interactions'] += 1
                    
                except Exception as e:
                    logger.error(f"Error in interactive learning: {e}")
                    print(f"❌ Error: {e}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interactive learning interrupted")

    def _show_system_insights(self):
        """PRESERVED: Display comprehensive system insights"""
        print("\n🧠 SYSTEM INSIGHTS")
        print("=" * 50)
        
        try:
            insights = self.system.get_system_insights()
            
            print("📊 TRIE STATISTICS:")
            trie_stats = insights.get('trie_statistics', {})
            print(f"  • Total sequences: {trie_stats.get('total_sequences', 0)}")
            print(f"  • Context window size: {trie_stats.get('context_window_size', 0)}")
            print(f"  • Current context available: {trie_stats.get('current_context_available', False)}")
            
            print("\n⚡ HIGH-ACTIVATION NODES:")
            high_activation = insights.get('high_performing_nodes', {}).get('high_activation', [])
            if high_activation:
                for i, node in enumerate(high_activation[:5], 1):
                    print(f"  {i}. '{node['token']}' - Activation: {node['activation_level']:.3f}, "
                          f"Access count: {node['access_count']}")
            else:
                print("  No high-activation nodes found")
            
            print("\n🏆 HIGH-REWARD NODES:")
            high_reward = insights.get('high_performing_nodes', {}).get('high_reward', [])
            if high_reward:
                for i, node in enumerate(high_reward[:5], 1):
                    print(f"  {i}. '{node['token']}' - Avg reward: {node['avg_reward']:.3f}, "
                          f"Total reward: {node['total_reward']:.3f}, Frequency: {node['frequency']}")
            else:
                print("  No high-reward nodes found")
            
            print("\n🔍 BEAM SEARCH CONFIG:")
            beam_config = insights.get('beam_search_config', {})
            print(f"  • Beam width: {beam_config.get('beam_width', 'N/A')}")
            print(f"  • Max generation length: {beam_config.get('max_generation_length', 'N/A')}")
            
            scoring_weights = beam_config.get('scoring_weights', {})
            if scoring_weights:
                print("  • Scoring weights:")
                for component, weight in scoring_weights.items():
                    print(f"    - {component.title()}: {weight:.2f}")
            
            identity = insights.get('identity_context', {})
            if identity:
                print(f"\n👤 IDENTITY CONTEXT:")
                for key, value in identity.items():
                    print(f"  • {key}: {value}")
            
            logger.info("Displayed comprehensive system insights")
            
        except Exception as e:
            logger.error(f"Error showing system insights: {e}")
            print(f"❌ Error retrieving insights: {e}")

    def _handle_hf_training(self):
        """PRESERVED: Handle training on HuggingFace datasets"""
        if not self.hf_integration:
            print("❌ HuggingFace integration not available.")
            return

        try:
            print("\n📚 HUGGINGFACE DATASET TRAINING (Progressive Prediction Mode)")
            print("Available datasets:")
            print("1. PersonaChat (conversational)")
            print("2. Daily Dialog (daily conversations)")
            print("3. WikiText-2 (encyclopedia text)")
            print("4. Reddit Dataset (reddit posts and comments)")
            print("5. Train on all datasets")

            choice = input("Select dataset (1-5): ").strip()

            max_samples = input("Max samples per dataset (default 1000): ").strip()
            max_samples = int(max_samples) if max_samples else 1000

            if choice in ["1", "2", "3", "4"]:
                if choice == "1":
                    count = self.hf_integration.process_dataset(
                        "bavard/personachat_truecased", None, "persona_chat", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed PersonaChat dataset: {count} samples")
                elif choice == "2":
                    count = self.hf_integration.process_dataset(
                        "daily_dialog", None, "daily_dialog", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed Daily Dialog dataset: {count} samples")
                elif choice == "3":
                    count = self.hf_integration.process_dataset(
                        "wikitext", "wikitext-2-raw-v1", "wikitext_2_raw", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed WikiText-2 dataset: {count} samples")
                elif choice == "4":
                    count = self.hf_integration.process_dataset(
                        "SocialGrep/the-reddit-dataset-dataset", "comments", "comments", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed Reddit dataset: {count} samples")
                
                if count:
                    self.session_stats['dataset_samples'] += count

            elif choice == "5":
                results = {}
                for dataset_name, config, friendly_name in self.hf_integration.dataset_configs:
                    try:
                        count = self.hf_integration.process_dataset(
                            dataset_name, config, friendly_name,
                            max_samples, predictor_instance=self, shuffle=True
                        )
                        results[friendly_name] = count
                        if count:
                            self.session_stats['dataset_samples'] += count
                    except Exception as e:
                        logger.error(f"Error processing {friendly_name}: {e}")
                        results[friendly_name] = 0

                print("\n✅ Training Results:")
                for dataset, count in results.items():
                    print(f"   {dataset}: {count if count is not None else 0} samples")
            else:
                print("❌ Invalid choice.")

        except Exception as e:
            logger.error(f"Error in HuggingFace training: {e}")
            print(f"❌ Training error: {e}")

    def _handle_statistics(self):
        """PRESERVED: Display comprehensive learning statistics"""
        print("\n📊 LEARNING STATISTICS")
        print("=" * 60)
        
        try:
            runtime = time.time() - self.session_stats['start_time']
            print("🕒 SESSION STATISTICS:")
            print(f"  • Runtime: {runtime:.1f} seconds")
            print(f"  • Interactions: {self.session_stats['interactions']}")
            print(f"  • Predictions Made: {self.session_stats['predictions']}")
            print(f"  • Dataset Samples: {self.session_stats['dataset_samples']}")
            
            insights = self.system.get_system_insights()
            
            print(f"\n💾 MEMORY STATISTICS:")
            trie_stats = insights.get('trie_statistics', {})
            print(f"  • Total sequences stored: {trie_stats.get('total_sequences', 0)}")
            print(f"  • Context window size: {trie_stats.get('context_window_size', 0)}")
            print(f"  • Context available: {trie_stats.get('current_context_available', False)}")
            
            print(f"\n⚡ ACTIVATION ANALYSIS:")
            high_activation = insights.get('high_performing_nodes', {}).get('high_activation', [])
            if high_activation:
                print(f"  • Top {len(high_activation)} most active nodes:")
                for i, node in enumerate(high_activation, 1):
                    print(f"    {i}. '{node['token']}' - Activation: {node['activation_level']:.3f}")
                
                avg_activation = sum(node['activation_level'] for node in high_activation) / len(high_activation)
                print(f"  • Average activation level: {avg_activation:.3f}")
            else:
                print("  • No high-activation nodes detected")
            
            print(f"\n🏆 REWARD ANALYSIS:")
            high_reward = insights.get('high_performing_nodes', {}).get('high_reward', [])
            if high_reward:
                print(f"  • Top {len(high_reward)} highest-reward nodes:")
                for i, node in enumerate(high_reward, 1):
                    print(f"    {i}. '{node['token']}' - Avg: {node['avg_reward']:.3f}, "
                          f"Total: {node['total_reward']:.3f}")
                
                avg_reward = sum(node['avg_reward'] for node in high_reward) / len(high_reward)
                total_reward = sum(node['total_reward'] for node in high_reward)
                print(f"  • Average reward: {avg_reward:.3f}")
                print(f"  • Total accumulated reward: {total_reward:.3f}")
            else:
                print("  • No high-reward nodes detected")
            
            print(f"\n🔍 PREDICTION SYSTEM:")
            beam_config = insights.get('beam_search_config', {})
            print(f"  • Beam width: {beam_config.get('beam_width', 'N/A')}")
            print(f"  • Max generation length: {beam_config.get('max_generation_length', 'N/A')}")
            
            scoring_weights = beam_config.get('scoring_weights', {})
            if scoring_weights:
                print("  • Scoring component weights:")
                for component, weight in scoring_weights.items():
                    print(f"    - {component.replace('_', ' ').title()}: {weight:.2f}")
            
            if self.session_stats['interactions'] > 0:
                print(f"\n📈 LEARNING EFFICIENCY:")
                predictions_per_interaction = self.session_stats['predictions'] / self.session_stats['interactions']
                print(f"  • Predictions per interaction: {predictions_per_interaction:.2f}")
                
                if runtime > 0:
                    interactions_per_minute = (self.session_stats['interactions'] / runtime) * 60
                    print(f"  • Interactions per minute: {interactions_per_minute:.1f}")
            
            identity = insights.get('identity_context', {})
            if identity:
                print(f"\n👤 IDENTITY CONTEXT:")
                for key, value in identity.items():
                    print(f"  • {key}: {value}")
            
            logger.info("Displayed comprehensive learning statistics")
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            print(f"❌ Error retrieving statistics: {e}")

    def _handle_save_load(self):
        """PRESERVED: Handle save/load model functionality"""
        print("\n💾 SAVE/LOAD MODEL")
        print("1. Save model")
        print("2. Load model")
        
        choice = input("Select option (1-2): ").strip()
        
        if choice == "1":
            try:
                print("✅ Model automatically saved to LMDB database")
                logger.info("Model save completed (LMDB handles persistence)")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                print(f"❌ Save error: {e}")
                
        elif choice == "2":
            try:
                print("✅ Model loads automatically from LMDB database on startup")
                logger.info("Model load completed (LMDB loads automatically)")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                print(f"❌ Load error: {e}")
        else:
            print("❌ Invalid choice")

    def run(self):
        """PRESERVED: Main interactive loop"""
        print("🚀 Starting Recursive Token Weaver System...")
        logger.info("Starting interactive user interface")
        
        try:
            while True:
                self.show_menu()
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    self._handle_interactive_learning()
                elif choice == "2":
                    self._handle_hf_training()
                elif choice == "3":
                    self._handle_statistics()
                elif choice == "4":
                    self._handle_save_load()
                elif choice == "5":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
        except KeyboardInterrupt:
            print("\n👋 System shutdown initiated")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"❌ System error: {e}")
        finally:
            try:
                self.system.close()
                logger.info("System shutdown completed")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

# Example usage and testing
if __name__ == "__main__":
    logger.info("Starting Enhanced Trie-Based Predictive System")
    
    try:
        print("🎯 Choose startup mode:")
        print("1. Interactive Menu System")
        print("2. Demo Mode (quick test)")
        
        mode = input("Enter choice (1-2): ").strip()
        
        if mode == "1":
            # Interactive menu system
            interface = UserInteraction()
            interface.run()
            
        elif mode == "2":
            # Demo mode - preserved original testing logic
            system = PredictiveSystem()
            
            training_data = [
                ("hello how are you", 0.8),
                ("hello there friend", 0.7),
                ("how are you doing", 0.9),
                ("i am doing well", 0.8),
                ("thank you very much", 0.9)
            ]
            
            print("Training the system...")
            for text, reward in training_data:
                result = system.process_input(text, reward)
                print(f"Processed: '{text}' -> ID: {result['sequence_id'][:8]}...")
            
            print("\nTesting predictions...")
            test_queries = ["hello", "how are", "i am"]
            
            for query in test_queries:
                continuation, confidence = system.predict_continuation(query)
                if continuation:
                    full_prediction = f"{''.join(continuation)}"
                    print(f"Query: '{query}' -> Prediction: '{full_prediction}' (confidence: {confidence:.3f})")
                else:
                    print(f"Query: '{query}' -> No prediction found")
            
            system.close()
            logger.info("Demo completed successfully")
        else:
            print("❌ Invalid choice")
        
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        raise