from TicTacToe_Env import TicTacToeEnv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

load_last_model = True

# Get the environment and extract the number of actions.
size = 3
np.random.seed(123)
nb_actions = size * size

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1, 9)))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

env = TicTacToeEnv(size=size, agent=dqn)

if load_last_model:
    try:
        # Load weights from last experiment
        dqn.load_weights('models/dqn_{}_weights.h5f'.format(env.name))
    except FileExistsError:
        print(FileExistsError)
    except Exception:
        print(Exception)

if __name__ == '__main__':
    selection = int(input('''
        Select action:
        1 - train model
        2 - play vs. model
    '''))
    if selection == 1:
        for _ in range(100):
            dqn.fit(env, nb_steps=10_000, visualize=False, verbose=1)

            # After training is done, we save the final weights.
            print('saving model')
            dqn.save_weights('models/dqn_{}_weights.h5f'.format(env.name), overwrite=True)

            # Finally, evaluate our algorithm for 5 episodes.
            # dqn.test(env, nb_episodes=5, visualize=False)
    else:
        env.reset()
        while True:
            env.render()
            player_move = int(input('your move: '))
            if env._legal_move_mask[player_move] == 1:
                _, _, done, _ = env.step(player_move)
                if done:
                    env.render()
                    input()
                    env.reset()
                    continue
