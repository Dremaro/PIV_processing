import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
y = np.sin(x)
line, = ax.plot(x, y)

def animate(i):
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

# Pour sauvegarder l'animation, utilisez e.g.
# ani.save("movie.mp4")

plt.show()