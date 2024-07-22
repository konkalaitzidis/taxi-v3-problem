# taxi-v3-problem
This repository contains a solution to the taxi-v3 toy problem, which is a classic reinforcement learning (RL) environment provided by OpenAI's Gym library. More info: https://gymnasium.farama.org/environments/toy_text/taxi/.

## Background
The solution to the taxi-v3 problem has been approached by applying the Q-learning algorithm instead of other alternatives such as SARSA (State-Action-Reward-State-Action).

Q-learning is an off-policy method that updates its Q-values based on the next state's reward and the consequent maximum reward of all the possible states after that, resulting in faster convergence to the optimal policy. It is more suitable for problems with deterministic environments with known rewards such as taxi-v3. On the other had, SARSA is more suitbale for stochastic environments where a more explorative approach is required.


<!-- Approach: -->

## Installation

To get started, follow these steps:

1. Clone the Repository. Open your terminal and navigate to a directory of your choice. Then clone the repository with:
    ```sh
    git clone https://github.com/konkalaitzidis/taxi-v3-problem.git
    ```
    ```sh
    cd taxi-v3-problem
    ```

2. Set up a Virtual Environment. Create and activate a Virtual Environment for managing dependencies (recommended). Replace `name_of_venv` with your desired environment name.

    Linux/macOS:
    ```sh
    python -m venv name_of_venv
    ```
    ```sh
    source name_of_venv/bin/activate 
    ```
    Windows: 
    ```sh
    python -m venv name_of_venv
    ```
    ```sh
    name_of_venv\Scripts\activate
    ```

3. Install Dependencies. Install the required packages by running the following command:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the solution, execute the following command:
```sh
python main.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.