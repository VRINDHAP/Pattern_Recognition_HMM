# Pattern Recognition Assignment: HMM Baum-Welch Algorithm

**Name:** [VRINDHA P]  
**University Register Number:** [TCR24CS071]  

## Description
This project implements the Baum-Welch algorithm for Hidden Markov Models (HMM) in Python. The implementation is based on the provided class PDF, utilizing the "Walk/Shop" observation sequence to guess the "Rainy/Sunny" hidden states.

The script calculates and outputs:
- Probability of each state transition (Transition Matrix A)
- Emission Matrix B
- Initial Distribution pi
- P(O | lambda)

It also visually graphs the probability changes over iterations and generates a state transition diagram.

## How to Run the Program
1. Make sure you have Python installed.
2. Install the required mathematical and graphing libraries by typing this in your terminal:
   `pip install numpy matplotlib graphviz`
3. Run the script:
   `python hmm.py`
4. The terminal will print the matrices. The program will also generate `probability_plots.png` and `hmm_state_diagram.png` in the folder.