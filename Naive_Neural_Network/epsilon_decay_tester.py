import matplotlib.pyplot as plt

epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.9985

games = 10_000
epsilon_over_time = []
for i in range(games):

    epsilon_over_time.append(epsilon)

    if epsilon > min_epsilon:
        epsilon = epsilon * epsilon_decay
    else:
        epsilon = min_epsilon


plt.plot(epsilon_over_time)
plt.show()