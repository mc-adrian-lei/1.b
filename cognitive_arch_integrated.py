# ============================================================================
# COGNITIVE ARCHITECTURE WITH INTEGRATED PATTERN RECOGNITION
# Integration of KnowledgeBase + PatternAwareSystem into CognitiveArchitecture
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from enum import Enum
import time
from abc import ABC, abstractmethod
import random
import difflib


# ============================================================================
# COMPONENT A: PATTERN RECOGNITION LAYER (Signal → Archetype Mapping)
# ============================================================================

class KnowledgeBase:
    """Stores known patterns (Archetypes) that the system recognizes."""
    def __init__(self):
        # Dictionary of Known Patterns: {Pattern_Name: Signature_String}
        # Each signature represents a canonical pattern of inputs/conflicts
        self.known_patterns = {
            "CRISIS_EVENT":     "1100110011",       # High urgency, immediate response needed
            "OPPORTUNITY":      "0011001100",       # Growth potential, balanced approach
            "SOCIAL_CONFLICT":  "1111000011",       # Interpersonal tension, requires diplomacy
            "ROUTINE_TASK":     "0000000000",       # Standard operation, low conflict
            "TRANSPARENCY_DEMAND": "1111111100",    # High disclosure pressure
            "COMPLIANCE_LOCK":  "0000001111",       # High restriction pressure
            "HYBRID_PRESSURE":  "1010101010"        # Balanced conflicting demands
        }
        
        # Archetype metadata linking patterns to pathway recommendations
        self.archetype_metadata = {
            "CRISIS_EVENT": {"primary_pathway": "hybrid", "urgency": "HIGH"},
            "OPPORTUNITY": {"primary_pathway": "transparency", "urgency": "MEDIUM"},
            "SOCIAL_CONFLICT": {"primary_pathway": "hybrid", "urgency": "MEDIUM"},
            "ROUTINE_TASK": {"primary_pathway": "non_disclosure", "urgency": "LOW"},
            "TRANSPARENCY_DEMAND": {"primary_pathway": "transparency", "urgency": "HIGH"},
            "COMPLIANCE_LOCK": {"primary_pathway": "non_disclosure", "urgency": "HIGH"},
            "HYBRID_PRESSURE": {"primary_pathway": "hybrid", "urgency": "CRITICAL"}
        }

    def get_highest_match(self, input_signal_signature: str) -> Tuple[str, float]:
        """
        Core Pattern Recognition Logic:
        Compares input signal against all known patterns 
        and returns the one with the highest similarity score (Resonance).
        """
        best_match = None
        highest_score = -1.0

        for name, pattern_sig in self.known_patterns.items():
            # Calculate Resonance (Similarity) using SequenceMatcher
            score = difflib.SequenceMatcher(None, input_signal_signature, pattern_sig).ratio()
            
            if score > highest_score:
                highest_score = score
                best_match = name
        
        return best_match, highest_score
    
    def get_archetype_recommendation(self, archetype_name: str) -> Dict:
        """Return metadata and routing recommendations for an archetype."""
        return self.archetype_metadata.get(archetype_name, {
            "primary_pathway": "hybrid",
            "urgency": "MEDIUM"
        })


# ============================================================================
# THREAD 1: STRUCTURAL LOGIC - Action Pathways (Original)
# ============================================================================

class PathwayType(Enum):
    """Enumeration of pathway types for classification."""
    TRANSPARENCY_REQUEST = "transparency_request"
    NON_DISCLOSURE = "non_disclosure"
    HYBRID_BALANCED = "hybrid_balanced"
    ABSTRACTION_FILTER = "abstraction_filter"


@dataclass
class ProcessTrace:
    """Records the structural decision points for a pathway."""
    intent: str
    conflict_type: str
    guardrail_status: str  # ACTIVE, PASS, or BLOCK
    action_taken: str
    result_state: str
    archetype_detected: str = "UNKNOWN"  # NEW: Archetype that triggered this trace


class Pathway(ABC):
    """Abstract base class for action pathways."""
    
    def __init__(self, name: str, pathway_type: PathwayType):
        self.name = name
        self.pathway_type = pathway_type
        self.process_trace = None
        self.input_count = 0
        self.success_count = 0
    
    @abstractmethod
    def evaluate_suitability(self, input_x: Dict) -> float:
        """
        Evaluate how suitable this pathway is for the given input.
        Returns a score between 0.0 and 1.0.
        """
        pass
    
    @abstractmethod
    def process(self, input_x: Dict) -> Dict:
        """
        Execute the pathway's processing logic.
        Returns output with processed data and metadata.
        """
        pass
    
    def create_trace(self, input_x: Dict, archetype: str = "UNKNOWN") -> ProcessTrace:
        """Create a structural trace of the processing decision."""
        return ProcessTrace(
            intent=input_x.get('intent', 'unknown'),
            conflict_type=input_x.get('conflict_type', 'none'),
            guardrail_status=input_x.get('guardrail_status', 'NEUTRAL'),
            action_taken=f"Pathway: {self.name}",
            result_state=f"Processing via {self.pathway_type.value}",
            archetype_detected=archetype
        )


class TransparencyPathway(Pathway):
    """Pathway optimized for transparency requests."""
    
    def __init__(self):
        super().__init__("TransparencyPathway", PathwayType.TRANSPARENCY_REQUEST)
        self.max_disclosure_depth = 5
    
    def evaluate_suitability(self, input_x: Dict) -> float:
        """High suitability for transparency-focused inputs."""
        if input_x.get('intent') == 'transparency_request':
            return 0.95
        elif input_x.get('conflict_type') == 'mixed':
            return 0.65
        else:
            return 0.2
    
    def process(self, input_x: Dict) -> Dict:
        """Generate maximum transparency within guardrail constraints."""
        self.input_count += 1
        archetype = input_x.get('detected_archetype', 'UNKNOWN')
        trace = self.create_trace(input_x, archetype)
        
        disclosure_level = min(self.max_disclosure_depth, input_x.get('depth', 3))
        output = {
            'pathway_name': self.name,
            'disclosure_level': disclosure_level,
            'transparency_score': min(1.0, disclosure_level / self.max_disclosure_depth),
            'content': f"Transparency output at depth {disclosure_level}",
            'trace': trace,
            'guardrail_maintained': True
        }
        
        self.success_count += 1
        return output


class NonDisclosurePathway(Pathway):
    """Pathway optimized for non-disclosure mandates."""
    
    def __init__(self):
        super().__init__("NonDisclosurePathway", PathwayType.NON_DISCLOSURE)
        self.abstraction_layers = 3
    
    def evaluate_suitability(self, input_x: Dict) -> float:
        """High suitability for non-disclosure-focused inputs."""
        if input_x.get('intent') == 'non_disclosure':
            return 0.95
        elif input_x.get('conflict_type') == 'mixed':
            return 0.65
        else:
            return 0.2
    
    def process(self, input_x: Dict) -> Dict:
        """Apply abstraction filtering while maintaining compliance."""
        self.input_count += 1
        archetype = input_x.get('detected_archetype', 'UNKNOWN')
        trace = self.create_trace(input_x, archetype)
        
        abstraction_depth = min(self.abstraction_layers, input_x.get('abstraction', 2))
        output = {
            'pathway_name': self.name,
            'abstraction_layers': abstraction_depth,
            'compliance_score': min(1.0, abstraction_depth / self.abstraction_layers),
            'content': f"Abstracted output with {abstraction_depth} filtering layers",
            'trace': trace,
            'guardrail_maintained': True
        }
        
        self.success_count += 1
        return output


class HybridBalancedPathway(Pathway):
    """Pathway balancing transparency and non-disclosure constraints."""
    
    def __init__(self):
        super().__init__("HybridPathway", PathwayType.HYBRID_BALANCED)
        self.balance_ratio = 0.5
    
    def evaluate_suitability(self, input_x: Dict) -> float:
        """High suitability for mixed or balanced inputs."""
        if input_x.get('conflict_type') == 'mixed':
            return 0.95
        elif input_x.get('intent') in ['transparency_request', 'non_disclosure']:
            return 0.5
        else:
            return 0.7
    
    def process(self, input_x: Dict) -> Dict:
        """Balance transparency and non-disclosure within constraints."""
        self.input_count += 1
        archetype = input_x.get('detected_archetype', 'UNKNOWN')
        trace = self.create_trace(input_x, archetype)
        
        transparency_level = self.balance_ratio * input_x.get('depth', 3)
        abstraction_level = (1 - self.balance_ratio) * input_x.get('abstraction', 2)
        
        output = {
            'pathway_name': self.name,
            'transparency_level': transparency_level,
            'abstraction_level': abstraction_level,
            'balance_score': 0.5,
            'content': f"Balanced output: transparency={transparency_level:.2f}, abstraction={abstraction_level:.2f}",
            'trace': trace,
            'guardrail_maintained': True
        }
        
        self.success_count += 1
        return output


# ============================================================================
# THREAD 2: METABOLIC COST - Resource-Aware Routing & Scheduling (Original)
# ============================================================================

@dataclass
class ResourceMetrics:
    """Tracks resource usage for a pathway."""
    compute_cost: float
    memory_cost: float
    latency_cost: float
    total_cost: float


class ResourceScheduler:
    """Manages resource allocation and cost tracking."""
    
    def __init__(self, vitality_initial: float = 100.0, decay_base: float = 1.5):
        self.vitality = vitality_initial
        self.decay_base = decay_base
        self.resource_logs = {}
        self.cost_models = {}
        self._initialize_cost_models()
    
    def _initialize_cost_models(self):
        """Define cost functions for each pathway type."""
        self.cost_models['TransparencyPathway'] = lambda depth: {
            'compute': 5.0 + (depth * 2.0),
            'memory': 2.0 + (depth * 0.5),
            'latency': 1.0 + (depth * 0.3)
        }
        self.cost_models['NonDisclosurePathway'] = lambda abstraction: {
            'compute': 1.0 + (abstraction * 1.5),
            'memory': 1.0 + (abstraction * 0.3),
            'latency': 0.5 + (abstraction * 0.2)
        }
        self.cost_models['HybridPathway'] = lambda params: {
            'compute': 3.0,
            'memory': 1.5,
            'latency': 0.8
        }
    
    def estimate_cost(self, pathway_name: str, input_x: Dict = None) -> float:
        """Estimate the metabolic cost (vitality delta) for executing a pathway."""
        if pathway_name not in self.cost_models:
            return 0.5
        
        if input_x is None:
            input_x = {}
        
        if 'TransparencyPathway' in pathway_name:
            depth = input_x.get('depth', 3)
            costs = self.cost_models[pathway_name](depth)
        elif 'NonDisclosurePathway' in pathway_name:
            abstraction = input_x.get('abstraction', 2)
            costs = self.cost_models[pathway_name](abstraction)
        else:
            costs = self.cost_models[pathway_name]({})
        
        total_cost = costs['compute'] + costs['memory'] + costs['latency']
        efficiency_score = 1.0 / (1.0 + total_cost)
        return efficiency_score
    
    def log_usage(self, pathway_name: str, input_x: Dict = None):
        """Log resource usage for a pathway execution."""
        if input_x is None:
            input_x = {}
        
        if pathway_name not in self.resource_logs:
            self.resource_logs[pathway_name] = []
        
        if 'TransparencyPathway' in pathway_name:
            depth = input_x.get('depth', 3)
            costs = self.cost_models[pathway_name](depth)
        elif 'NonDisclosurePathway' in pathway_name:
            abstraction = input_x.get('abstraction', 2)
            costs = self.cost_models[pathway_name](abstraction)
        else:
            costs = self.cost_models[pathway_name]({})
        
        total_cost = costs['compute'] + costs['memory'] + costs['latency']
        
        metrics = ResourceMetrics(
            compute_cost=costs['compute'],
            memory_cost=costs['memory'],
            latency_cost=costs['latency'],
            total_cost=total_cost
        )
        
        self.resource_logs[pathway_name].append(metrics)
        decay_factor = self.decay_base * (total_cost / 10.0)
        self.vitality -= decay_factor
    
    def get_vitality_status(self) -> Dict:
        """Return current vitality and resource state."""
        return {
            'vitality': max(0.0, self.vitality),
            'decay_rate': self.decay_base,
            'total_pathways_used': len(self.resource_logs),
            'total_executions': sum(len(logs) for logs in self.resource_logs.values())
        }
    
    def get_pathway_efficiency(self, pathway_name: str) -> float:
        """Calculate average efficiency for a pathway."""
        if pathway_name not in self.resource_logs or not self.resource_logs[pathway_name]:
            return 0.5
        
        avg_cost = sum(m.total_cost for m in self.resource_logs[pathway_name]) / len(self.resource_logs[pathway_name])
        efficiency = 1.0 / (1.0 + avg_cost)
        return efficiency


# ============================================================================
# THREAD 3: SOCIAL IMPACT - Social Scoring & Reputation (Original)
# ============================================================================

@dataclass
class ReputationState:
    """Tracks reputation and external relational outcomes."""
    status: str
    trust_score: float
    compliance_score: float
    transparency_score: float
    last_behavior_tag: str


class SocialScorer:
    """Evaluates social impact and manages reputation scoring."""
    
    def __init__(self):
        self.reputation_status = "NEUTRAL"
        self.trust_score = 0.5
        self.compliance_score = 0.9
        self.transparency_score = 0.5
        self.pathway_feedback = {}
        self.last_behavior_tag = "NEUTRAL"
    
    def evaluate(self, pathway_name: str, input_x: Dict) -> float:
        """Evaluate the social impact score for a pathway."""
        if 'Transparency' in pathway_name:
            base_score = self.transparency_score
        elif 'NonDisclosure' in pathway_name:
            base_score = self.compliance_score
        else:
            base_score = (self.compliance_score + self.transparency_score) / 2.0
        
        if input_x.get('conflict_type') == 'mixed':
            if 'Hybrid' in pathway_name:
                base_score *= 1.2
            else:
                base_score *= 0.9
        
        trust_factor = self.trust_score * 0.3
        social_score = (base_score * 0.7) + trust_factor
        
        return min(1.0, social_score)
    
    def update_feedback(self, pathway_name: str, input_x: Dict, output: Dict):
        """Update social scores based on pathway execution and output."""
        if pathway_name not in self.pathway_feedback:
            self.pathway_feedback[pathway_name] = []
        
        feedback_score = output.get('transparency_score', 0.5) if 'transparency_score' in output else 0.5
        if 'compliance_score' in output:
            feedback_score = output['compliance_score']
        
        self.pathway_feedback[pathway_name].append(feedback_score)
        
        avg_feedback = sum(self.pathway_feedback[pathway_name]) / len(self.pathway_feedback[pathway_name])
        self.trust_score = min(1.0, self.trust_score + (avg_feedback - 0.5) * 0.05)
        
        if avg_feedback > 0.8:
            self.reputation_status = "CONSTRUCTIVE"
            self.last_behavior_tag = "GENERATIVE"
        elif avg_feedback > 0.6:
            self.reputation_status = "TESTING"
            self.last_behavior_tag = "DISCIPLINED"
        else:
            self.reputation_status = "NEUTRAL"
            self.last_behavior_tag = "CAUTIOUS"
    
    def get_reputation_state(self) -> ReputationState:
        """Return current reputation state."""
        return ReputationState(
            status=self.reputation_status,
            trust_score=self.trust_score,
            compliance_score=self.compliance_score,
            transparency_score=self.transparency_score,
            last_behavior_tag=self.last_behavior_tag
        )
    
    def get_pathway_social_score(self, pathway_name: str) -> float:
        """Get average social score for a pathway."""
        if pathway_name not in self.pathway_feedback or not self.pathway_feedback[pathway_name]:
            return 0.5
        return sum(self.pathway_feedback[pathway_name]) / len(self.pathway_feedback[pathway_name])


# ============================================================================
# INTEGRATED ARCHITECTURE: Three Threads + Pattern Recognition
# ============================================================================

class CognitiveArchitecture:
    """
    Main cognitive architecture that orchestrates:
    - Pattern Recognition (Signal → Archetype)
    - Three-Threaded Braiding (Structural, Metabolic, Social)
    - Pathway Selection and Execution
    """
    
    def __init__(self):
        # NEW: Pattern Recognition Layer
        self.knowledge_base = KnowledgeBase()
        self.confidence_threshold = 0.65
        
        # Original Three Threads
        self.pathways = {
            'transparency': TransparencyPathway(),
            'non_disclosure': NonDisclosurePathway(),
            'hybrid': HybridBalancedPathway()
        }
        
        self.resource_scheduler = ResourceScheduler(vitality_initial=100.0, decay_base=1.5)
        self.social_scorer = SocialScorer()
        
        self.execution_history = []
    
    def recognize_archetype(self, input_signal: str) -> Tuple[str, float, Dict]:
        """
        NEW: Pattern Recognition Phase
        Maps incoming signal to known archetype using KnowledgeBase.
        Returns: (archetype_name, resonance_score, archetype_metadata)
        """
        archetype, resonance = self.knowledge_base.get_highest_match(input_signal)
        metadata = self.knowledge_base.get_archetype_recommendation(archetype)
        
        return archetype, resonance, metadata
    
    def enrich_input(self, raw_input: Dict, signal_signature: str) -> Dict:
        """
        NEW: Signal Enrichment Phase
        Augments raw input with detected archetype and routing suggestions.
        """
        archetype, resonance, metadata = self.recognize_archetype(signal_signature)
        
        # Enrich the input with archetype information
        enriched_input = raw_input.copy()
        enriched_input['detected_archetype'] = archetype
        enriched_input['archetype_resonance'] = resonance
        enriched_input['archetype_metadata'] = metadata
        
        # If high confidence, suggest primary pathway
        if resonance >= self.confidence_threshold:
            enriched_input['suggested_pathway'] = metadata.get('primary_pathway', 'hybrid')
            enriched_input['urgency_level'] = metadata.get('urgency', 'MEDIUM')
        
        return enriched_input
    
    def select_pathway(self, input_x: Dict) -> Tuple[str, float, Dict]:
        """
        Select the optimal pathway using three-threaded braiding.
        Now also considers archetype-based routing suggestions.
        """
        scores = {}
        detailed_scores = {}
        
        for pathway_name, pathway in self.pathways.items():
            # THREAD 1: STRUCTURAL LOGIC
            suitability = pathway.evaluate_suitability(input_x)
            
            # Apply archetype boost if this is the suggested pathway
            if input_x.get('suggested_pathway') == pathway_name:
                suitability *= 1.15  # 15% boost for archetype-aligned pathway
            
            # THREAD 2: METABOLIC COST
            resource_efficiency = self.resource_scheduler.estimate_cost(pathway_name, input_x)
            
            # THREAD 3: SOCIAL IMPACT
            social_score = self.social_scorer.evaluate(pathway_name, input_x)
            
            # BRAID: Combine three threads
            combined_score = (suitability * 0.4) + (resource_efficiency * 0.35) + (social_score * 0.25)
            
            scores[pathway_name] = combined_score
            detailed_scores[pathway_name] = {
                'structural_logic': suitability,
                'metabolic_cost': resource_efficiency,
                'social_impact': social_score,
                'combined': combined_score
            }
        
        best_pathway = max(scores, key=scores.get)
        best_score = scores[best_pathway]
        
        return best_pathway, best_score, detailed_scores
    
    def execute(self, raw_input: Dict, signal_signature: str = None) -> Dict:
        """
        Main execution flow:
        1. Pattern Recognition (signal → archetype)
        2. Input Enrichment
        3. Pathway Selection (three-threaded braiding)
        4. Processing
        5. Feedback & Reputation Update
        """
        
        # PHASE 1: PATTERN RECOGNITION
        if signal_signature is None:
            signal_signature = raw_input.get('signal_signature', '0000000000')
        
        archetype, resonance, metadata = self.recognize_archetype(signal_signature)
        
        # PHASE 2: INPUT ENRICHMENT
        enriched_input = self.enrich_input(raw_input, signal_signature)
        
        # PHASE 3: PATHWAY SELECTION (Three-threaded braiding)
        pathway_name, combined_score, detailed_scores = self.select_pathway(enriched_input)
        pathway = self.pathways[pathway_name]
        
        # PHASE 4: EXECUTION
        output = pathway.process(enriched_input)
        
        # PHASE 5: METABOLIC & SOCIAL UPDATES
        self.resource_scheduler.log_usage(pathway_name, enriched_input)
        vitality_status = self.resource_scheduler.get_vitality_status()
        
        self.social_scorer.update_feedback(pathway_name, enriched_input, output)
        reputation_state = self.social_scorer.get_reputation_state()
        
        # Construct execution record with pattern recognition metadata
        execution_record = {
            'timestamp': time.time(),
            'pattern_recognition': {
                'detected_archetype': archetype,
                'resonance_score': resonance,
                'archetype_metadata': metadata
            },
            'input': enriched_input,
            'selected_pathway': pathway_name,
            'selection_scores': detailed_scores,
            'combined_score': combined_score,
            'output': output,
            'vitality_status': vitality_status,
            'reputation_state': {
                'status': reputation_state.status,
                'trust_score': reputation_state.trust_score,
                'compliance_score': reputation_state.compliance_score,
                'transparency_score': reputation_state.transparency_score,
                'last_behavior_tag': reputation_state.last_behavior_tag
            },
            'three_thread_braiding': {
                'structural_logic': output.get('trace'),
                'metabolic_cost': {
                    'vitality': vitality_status['vitality'],
                    'efficiency': self.resource_scheduler.get_pathway_efficiency(pathway_name)
                },
                'social_impact': {
                    'reputation': reputation_state.status,
                    'social_score': self.social_scorer.get_pathway_social_score(pathway_name)
                }
            }
        }
        
        self.execution_history.append(execution_record)
        return execution_record
    
    def print_execution_summary(self, execution_record: Dict):
        """Pretty-print execution summary with pattern recognition info."""
        print("\n" + "="*80)
        print("COGNITIVE ARCHITECTURE EXECUTION SUMMARY")
        print("="*80)
        
        # PATTERN RECOGNITION RESULTS
        pr = execution_record['pattern_recognition']
        print(f"\nPATTERN RECOGNITION (Thread 0: Sensory Input):")
        print(f"  Detected Archetype: {pr['detected_archetype']}")
        print(f"  Resonance Score: {pr['resonance_score']:.3f}")
        print(f"  Urgency Level: {pr['archetype_metadata'].get('urgency', 'N/A')}")
        print(f"  Suggested Pathway: {pr['archetype_metadata'].get('primary_pathway', 'N/A')}")
        
        print(f"\nINPUT (enriched with archetype):")
        print(f"  Intent: {execution_record['input'].get('intent', 'N/A')}")
        print(f"  Conflict Type: {execution_record['input'].get('conflict_type', 'N/A')}")
        print(f"  Detected Archetype: {execution_record['input'].get('detected_archetype', 'N/A')}")
        
        print(f"\nPATHWAY SELECTION (Three-Threaded Braiding):")
        for pathway_name, scores in execution_record['selection_scores'].items():
            print(f"\n  {pathway_name}:")
            print(f"    Thread 1 - Structural Logic (Suitability): {scores['structural_logic']:.3f}")
            print(f"    Thread 2 - Metabolic Cost (Efficiency): {scores['metabolic_cost']:.3f}")
            print(f"    Thread 3 - Social Impact (Trust/Compliance): {scores['social_impact']:.3f}")
            print(f"    → Combined Score: {scores['combined']:.3f}")
        
        print(f"\nSELECTED PATHWAY: {execution_record['selected_pathway']}")
        print(f"Combined Score: {execution_record['combined_score']:.3f}")
        
        print(f"\nOUTPUT CONTENT:")
        print(f"  {execution_record['output'].get('content', 'N/A')}")
        
        print(f"\nMETABOLIC STATUS (Thread 2):")
        vitality = execution_record['vitality_status']
        print(f"  Vitality: {vitality['vitality']:.2f}")
        print(f"  Total Executions: {vitality['total_executions']}")
        
        print(f"\nSOCIAL STATUS (Thread 3):")
        rep = execution_record['reputation_state']
        print(f"  Reputation: {rep['status']}")
        print(f"  Trust Score: {rep['trust_score']:.3f}")
        
        print("\n" + "="*80)


# ============================================================================
# DEMONSTRATION: Integrated Architecture with Pattern Recognition
# ============================================================================

if __name__ == "__main__":
    print("Initializing Cognitive Architecture with Pattern Recognition...")
    arch = CognitiveArchitecture()
    
    # Test cases with signal signatures
    test_cases = [
        {
            'raw_input': {
                'intent': 'transparency_request',
                'conflict_type': 'transparency_focused',
                'depth': 4,
                'abstraction': 1,
                'guardrail_status': 'ACTIVE'
            },
            'signal': '1111111000'  # Close to TRANSPARENCY_DEMAND
        },
        {
            'raw_input': {
                'intent': 'emergency_response',
                'conflict_type': 'crisis',
                'depth': 2,
                'abstraction': 2,
                'guardrail_status': 'ACTIVE'
            },
            'signal': '1100110011'  # Close to CRISIS_EVENT
        },
        {
            'raw_input': {
                'intent': 'balanced_resolution',
                'conflict_type': 'mixed',
                'depth': 2,
                'abstraction': 2,
                'guardrail_status': 'ACTIVE'
            },
            'signal': '1010101010'  # Close to HYBRID_PRESSURE
        }
    ]
    
    print("\nExecuting test cases with Pattern Recognition + Three-Threaded Braiding...\n")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i} ---")
        execution_record = arch.execute(test_case['raw_input'], test_case['signal'])
        arch.print_execution_summary(execution_record)
    
    print("\n" + "="*80)
    print("FINAL ARCHITECTURE STATE")
    print("="*80)
    print(f"\nVitality: {arch.resource_scheduler.get_vitality_status()['vitality']:.2f}")
    print(f"Reputation: {arch.social_scorer.get_reputation_state().status}")
    print(f"Trust: {arch.social_scorer.get_reputation_state().trust_score:.3f}")
    print(f"Total Executions: {len(arch.execution_history)}")
