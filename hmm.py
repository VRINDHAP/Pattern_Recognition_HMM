import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the local state-transition-diagrams library to the path so Python can find it
sys.path.append(os.path.join(os.path.dirname(__file__), 'state-transition-diagrams', 'State-Transition-Diagrams-e6f938b508b6246a7b859862a6e36ab25015166e'))
from state_transition_diagrams import render_state_diagram, DiagramConfig

def baum_welch(O, N, M, iterations=10):
    T = len(O)
    
    # Initial Parameters matching the PDF setup
    # pi = Initial probabilities (Rainy: 0.6, Sunny: 0.4)
    pi = np.array([0.6, 0.4]) 
    
    # A = Transition Matrix [Rainy->Rainy, Rainy->Sunny], [Sunny->Rainy, Sunny->Sunny]
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    
    # B = Emission Matrix (Rows: Rainy, Sunny. Cols: Walk, Shop)
    B = np.array([[0.1, 0.9], [0.6, 0.4]])
    
    p_o_history = []
    
    for it in range(iterations):
        # 1. Forward Pass (Alpha)
        alpha = np.zeros((T, N))
        for i in range(N):
            alpha[0, i] = pi[i] * B[i, O[0]]
        for t in range(1, T):
            for i in range(N):
                alpha[t, i] = np.sum(alpha[t-1, :] * A[:, i]) * B[i, O[t]]
                
        # 2. Backward Pass (Beta)
        beta = np.zeros((T, N))
        beta[T-1, :] = 1
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i, :] * B[:, O[t+1]] * beta[t+1, :])
                
        # 3. Probability of Observation Sequence P(O | lambda)
        P_O = np.sum(alpha[T-1, :])
        p_o_history.append(P_O)
        
        # 4. Gamma and Xi (Responsibilities)
        gamma = np.zeros((T, N))
        xi = np.zeros((T-1, N, N))
        for t in range(T):
            gamma[t, :] = (alpha[t, :] * beta[t, :]) / P_O
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, O[t+1]] * beta[t+1, j]) / P_O
                    
        # 5. Update Parameters
        pi = gamma[0, :]
        for i in range(N):
            gamma_sum = np.sum(gamma[:-1, i])
            for j in range(N):
                A[i, j] = np.sum(xi[:, i, j]) / gamma_sum
        for i in range(N):
            gamma_sum_T = np.sum(gamma[:, i])
            for k in range(M):
                mask = (np.array(O) == k)
                B[i, k] = np.sum(gamma[mask, i]) / gamma_sum_T
                
    return pi, A, B, p_o_history

if __name__ == "__main__":
    # From PDF: Walk (W) = 0, Shop (H) = 1. Rainy (R) = State 0, Sunny (S) = State 1
    # Long Sequence from PDF: W, H, H, W, H  -> 0, 1, 1, 0, 1
    observed_sequence = [0, 1, 1, 0, 1] 
    num_hidden_states = 2
    num_observation_symbols = 2
    
    print("Running Baum-Welch Algorithm...")
    pi, A, B, p_o_history = baum_welch(observed_sequence, num_hidden_states, num_observation_symbols, iterations=10)
    
    print("\n--- Final Outputs ---")
    print(f"Initial Distribution (pi):\n{pi}")
    print(f"\nTransition Matrix (A):\n{A}")
    print(f"\nEmission Matrix (B):\n{B}")
    
    # --- Plotting ---
    iterations_list = range(1, len(p_o_history) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations_list, p_o_history, marker='o', color='blue')
    plt.title("P(O | λ) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    one_minus_p = [1 - p for p in p_o_history]
    plt.plot(iterations_list, one_minus_p, marker='s', color='red')
    plt.title("1 - P(O | λ) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("1 - Probability")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("probability_plots.png")
    print("\nGraphs saved as 'probability_plots.png'.")
    
    # --- Generate State Transition Diagram ---
    try:
        config = DiagramConfig(title="HMM Transition Diagram", output_format="png", prob_threshold=0.01)
        labels = ["Rainy", "Sunny"]
        render_state_diagram(A, state_labels=labels, config=config, save_path="hmm_state_diagram")
        print("State transition diagram saved as 'hmm_state_diagram.png'.")
    except Exception as e:
        print(f"Could not generate state diagram due to error: {e}")
        print("Please ensure graphviz is installed on your system.")