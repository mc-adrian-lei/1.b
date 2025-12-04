# Cognitive Architecture: Executable Implementation
# Three-Threaded Braiding: Structural Logic, Metabolic Cost, Social Impact

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from enum import Enum
import time
from abc import ABC, abstractmethod


# ============================================================================
# THREAD 1: STRUCTURAL LOGIC - Action Pathways
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
    
    def create_trace(self, input_x: Dict) -> ProcessTrace:
        """Create a structural trace of the processing decision."""
        return ProcessTrace(
            intent=input_x.get('intent', 'unknown'),
            conflict_type=input_x.get('conflict_type', 'none'),
            guardrail_status=input_x.get('guardrail_status', 'NEUTRAL'),
            action_taken=f"Pathway: {self.name}",
            result_state=f"Processing via {self.pathway_type.value}"
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
        trace = self.create_trace(input_x)
        
        # Simulate transparency logic
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
        trace = self.create_trace(input_x)
        
        # Simulate abstraction logic
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
        self.balance_ratio = 0.5  # 50/50 transparency vs. abstraction
    
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
        trace = self.create_trace(input_x)
        
        # Simulate hybrid logic
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
# THREAD 2: METABOLIC COST - Resource-Aware Routing & Scheduling
# ============================================================================

@dataclass
class ResourceMetrics:
    """Tracks resource usage for a pathway."""
    compute_cost: float  # CPU cycles or normalized compute
    memory_cost: float   # Memory allocation
    latency_cost: float  # Processing time
    total_cost: float    # Aggregate cost


class ResourceScheduler:
    """Manages resource allocation and cost tracking."""
    
    def __init__(self, vitality_initial: float = 100.0, decay_base: float = 1.5):
        self.vitality = vitality_initial
        self.decay_base = decay_base
        self.resource_logs = {}  # pathway_name -> List[ResourceMetrics]
        self.cost_models = {}    # pathway_name -> cost function
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
        """
        Estimate the metabolic cost (vitality delta) for executing a pathway.
        Returns a score between 0.0 and 1.0 (lower cost = higher score).
        """
        if pathway_name not in self.cost_models:
            return 0.5  # Default neutral cost
        
        # Get cost model parameters from input
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
        
        # Normalize to 0.0-1.0 range (lower cost = higher efficiency score)
        efficiency_score = 1.0 / (1.0 + total_cost)
        return efficiency_score
    
    def log_usage(self, pathway_name: str, input_x: Dict = None):
        """Log resource usage for a pathway execution."""
        if input_x is None:
            input_x = {}
        
        if pathway_name not in self.resource_logs:
            self.resource_logs[pathway_name] = []
        
        # Calculate costs
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
        
        # Update vitality (metabolic state)
        decay_factor = self.decay_base * (total_cost / 10.0)  # Normalize decay
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
# THREAD 3: SOCIAL IMPACT - Social Scoring & Reputation
# ============================================================================

@dataclass
class ReputationState:
    """Tracks reputation and external relational outcomes."""
    status: str  # NEUTRAL, TESTING, CONSTRUCTIVE, GENERATIVE
    trust_score: float  # 0.0-1.0
    compliance_score: float  # 0.0-1.0
    transparency_score: float  # 0.0-1.0
    last_behavior_tag: str


class SocialScorer:
    """Evaluates social impact and manages reputation scoring."""
    
    def __init__(self):
        self.reputation_status = "NEUTRAL"
        self.trust_score = 0.5
        self.compliance_score = 0.9  # High compliance by default
        self.transparency_score = 0.5  # Balanced initially
        self.pathway_feedback = {}  # pathway_name -> List[scores]
        self.last_behavior_tag = "NEUTRAL"
    
    def evaluate(self, pathway_name: str, input_x: Dict) -> float:
        """
        Evaluate the social impact score for a pathway.
        Incorporates trust, compliance, and transparency metrics.
        Returns a score between 0.0 and 1.0.
        """
        # Base score from pathway type
        if 'Transparency' in pathway_name:
            base_score = self.transparency_score
        elif 'NonDisclosure' in pathway_name:
            base_score = self.compliance_score
        else:
            base_score = (self.compliance_score + self.transparency_score) / 2.0
        
        # Adjust based on input context
        if input_x.get('conflict_type') == 'mixed':
            # Hybrid pathways score higher for mixed conflicts
            if 'Hybrid' in pathway_name:
                base_score *= 1.2
            else:
                base_score *= 0.9
        
        # Incorporate trust
        trust_factor = self.trust_score * 0.3  # Trust contributes 30%
        social_score = (base_score * 0.7) + trust_factor
        
        return min(1.0, social_score)  # Clamp to [0.0, 1.0]
    
    def update_feedback(self, pathway_name: str, input_x: Dict, output: Dict):
        """
        Update social scores based on pathway execution and output.
        Simulates user feedback and behavioral adaptation.
        """
        if pathway_name not in self.pathway_feedback:
            self.pathway_feedback[pathway_name] = []
        
        # Simulate feedback score (0.0-1.0)
        feedback_score = output.get('transparency_score', 0.5) if 'transparency_score' in output else 0.5
        if 'compliance_score' in output:
            feedback_score = output['compliance_score']
        
        self.pathway_feedback[pathway_name].append(feedback_score)
        
        # Update reputation based on cumulative feedback
        avg_feedback = sum(self.pathway_feedback[pathway_name]) / len(self.pathway_feedback[pathway_name])
        
        # Update trust
        self.trust_score = min(1.0, self.trust_score + (avg_feedback - 0.5) * 0.05)
        
        # Update reputation status
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
# MAIN: Cognitive Architecture with Three-Threaded Braiding
# ============================================================================

class CognitiveArchitecture:
    """
    Main cognitive architecture that orchestrates the three threads:
    - Structural Logic (pathways and decision points)
    - Metabolic Cost (resource scheduling and vitality)
    - Social Impact (reputation and social scoring)
    """
    
    def __init__(self):
        # Initialize pathways
        self.pathways = {
            'transparency': TransparencyPathway(),
            'non_disclosure': NonDisclosurePathway(),
            'hybrid': HybridBalancedPathway()
        }
        
        # Initialize resource and social components
        self.resource_scheduler = ResourceScheduler(vitality_initial=100.0, decay_base=1.5)
        self.social_scorer = SocialScorer()
        
        # Execution history
        self.execution_history = []
    
    def select_pathway(self, input_x: Dict) -> Tuple[str, float]:
        """
        Select the optimal pathway using three-threaded braiding.
        
        Returns:
            (pathway_name, combined_score)
        
        The selection process braids three threads:
        1. STRUCTURAL LOGIC: Suitability of pathway for input
        2. METABOLIC COST: Efficiency of resource usage
        3. SOCIAL IMPACT: Reputation and trust metrics
        """
        scores = {}
        detailed_scores = {}
        
        for pathway_name, pathway in self.pathways.items():
            # THREAD 1: STRUCTURAL LOGIC
            suitability = pathway.evaluate_suitability(input_x)
            
            # THREAD 2: METABOLIC COST
            resource_efficiency = self.resource_scheduler.estimate_cost(pathway_name, input_x)
            
            # THREAD 3: SOCIAL IMPACT
            social_score = self.social_scorer.evaluate(pathway_name, input_x)
            
            # BRAID: Combine three threads into unified selection score
            combined_score = (suitability * 0.4) + (resource_efficiency * 0.35) + (social_score * 0.25)
            
            scores[pathway_name] = combined_score
            detailed_scores[pathway_name] = {
                'structural_logic': suitability,
                'metabolic_cost': resource_efficiency,
                'social_impact': social_score,
                'combined': combined_score
            }
        
        # Select pathway with highest combined score
        best_pathway = max(scores, key=scores.get)
        best_score = scores[best_pathway]
        
        return best_pathway, best_score, detailed_scores
    
    def execute(self, input_x: Dict) -> Dict:
        """
        Execute the architecture: select pathway, process, and update metrics.
        
        This is where the three threads actively braid during execution:
        - Structural Logic captures the decision trace
        - Metabolic Cost tracks resource usage
        - Social Impact scores reputation and learns from feedback
        """
        # SELECTION PHASE: Three-threaded pathway selection
        pathway_name, combined_score, detailed_scores = self.select_pathway(input_x)
        pathway = self.pathways[pathway_name]
        
        # EXECUTION PHASE: Process through selected pathway
        output = pathway.process(input_x)
        
        # METABOLIC PHASE: Log resource usage (Thread 2)
        self.resource_scheduler.log_usage(pathway_name, input_x)
        vitality_status = self.resource_scheduler.get_vitality_status()
        
        # SOCIAL PHASE: Update reputation and feedback (Thread 3)
        self.social_scorer.update_feedback(pathway_name, input_x, output)
        reputation_state = self.social_scorer.get_reputation_state()
        
        # Construct comprehensive execution record
        execution_record = {
            'timestamp': time.time(),
            'input': input_x,
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
        """Pretty-print execution summary showing three-threaded braiding."""
        print("\n" + "="*80)
        print("COGNITIVE ARCHITECTURE EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\nINPUT:")
        print(f"  Intent: {execution_record['input'].get('intent', 'N/A')}")
        print(f"  Conflict Type: {execution_record['input'].get('conflict_type', 'N/A')}")
        
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
        print(f"  Decay Rate: {vitality['decay_rate']}")
        print(f"  Total Executions: {vitality['total_executions']}")
        
        print(f"\nSOCIAL STATUS (Thread 3):")
        rep = execution_record['reputation_state']
        print(f"  Reputation: {rep['status']}")
        print(f"  Trust Score: {rep['trust_score']:.3f}")
        print(f"  Compliance Score: {rep['compliance_score']:.3f}")
        print(f"  Transparency Score: {rep['transparency_score']:.3f}")
        print(f"  Last Behavior: {rep['last_behavior_tag']}")
        
        print("\n" + "="*80)


# ============================================================================
# DEMONSTRATION: Three-Threaded Braiding in Action
# ============================================================================

if __name__ == "__main__":
    print("Initializing Cognitive Architecture with Three-Threaded Braiding...")
    arch = CognitiveArchitecture()
    
    # Test cases demonstrating different conflict scenarios
    test_cases = [
        {
            'intent': 'transparency_request',
            'conflict_type': 'transparency_focused',
            'depth': 4,
            'abstraction': 1,
            'guardrail_status': 'ACTIVE'
        },
        {
            'intent': 'non_disclosure',
            'conflict_type': 'compliance_focused',
            'depth': 1,
            'abstraction': 3,
            'guardrail_status': 'ACTIVE'
        },
        {
            'intent': 'balanced_resolution',
            'conflict_type': 'mixed',
            'depth': 2,
            'abstraction': 2,
            'guardrail_status': 'ACTIVE'
        },
        {
            'intent': 'transparency_request',
            'conflict_type': 'mixed',
            'depth': 3,
            'abstraction': 2,
            'guardrail_status': 'ACTIVE'
        }
    ]
    
    print("\nExecuting test cases with three-threaded braiding...\n")
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i} ---")
        execution_record = arch.execute(test_input)
        arch.print_execution_summary(execution_record)
    
    print("\n" + "="*80)
    print("FINAL ARCHITECTURE STATE")
    print("="*80)
    print(f"\nVitality: {arch.resource_scheduler.get_vitality_status()['vitality']:.2f}")
    print(f"Reputation: {arch.social_scorer.get_reputation_state().status}")
    print(f"Trust: {arch.social_scorer.get_reputation_state().trust_score:.3f}")
    print(f"\nTotal Executions: {len(arch.execution_history)}")
